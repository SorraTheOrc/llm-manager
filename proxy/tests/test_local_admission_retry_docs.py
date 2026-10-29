"""Docs contract: the per-reason retry table must match the implementation.

LP-0MVQ23BH7006HLKB (AC1–AC4 of that item; AC5 of parent LP-0MVNTT4X8009IO2X).
The deliverable here *is* the prose in ``proxy/docs/routing.md``, so a plain
"the words are present" assertion would only pin a copy of the table. Instead
the documented table is parsed and every row is replayed through the
**production** decision (:func:`provider._refusal_retry_timing`) and the
**production** header builder (:func:`provider._refusal_timing_headers`):

- the documented reason set is exactly the vocabulary the proxy emits, and the
  documented "no timer" rows are exactly
  :data:`provider.NOT_TIME_RETRYABLE_REFUSAL_REASONS` — a doc row for a reason
  that cannot happen (or a reason missing from the table) fails;
- each documented ``Retry-After`` / ``retryable`` / ``retry_at`` cell must
  match what the real decision produces for that reason, so a change to the
  timing logic cannot silently leave the docs lying;
- every header the refusal path actually emits must be documented, so a newly
  added header is caught as doc drift;
- ``retry_at`` is documented (and produced) as RFC 3339 UTC, and a non-admission
  expiry (a cooldown remainder) is documented as the *real* expiry rather than
  the ETA formula.

Content-based assertions remain only where the contract *is* prose: the
``local-preferred`` "pinned while deferrable, promoted when too large"
statement, which no production helper can express (LP-0MVSHXU8E004BL57 AC6).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from proxy import provider

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"

#: Heading of the documented contract; the section ends at the next heading.
SECTION_HEADING = "### Per-reason retry contract (LP-0MVNTT4X8009IO2X)"

#: The refusal reasons the proxy emits, as pinned by the cross-reason contract
#: table in ``test_local_admission_guard.py``. Documented in routing.md.
EMITTED_REFUSAL_REASONS = {
    "eta_exceeds_budget",
    "eta_exceeds_budget_direct",
    "prefill_eta_exceeds_deadline",
    "prefill_eta_exceeds_safety_budget",
    "prefill_budget_exceeded",
    "context_too_large",
    "cooldown",
}

#: Occupancy/ETA reasons whose timer is derived from the projected wait.
WAIT_BUDGET_REASONS = {
    "eta_exceeds_budget",
    "eta_exceeds_budget_direct",
    "prefill_eta_exceeds_deadline",
    "prefill_eta_exceeds_safety_budget",
}

#: Reasons whose timer comes from real expiry state rather than the ETA formula.
REAL_EXPIRY_REASONS = {"cooldown"}

#: Real cooldown remainder used to replay a documented non-admission row.
COOLDOWN_REMAINDER_SECONDS = 90
#: Projected wait / budget used to replay the documented ETA rows.
PROJECTED_WAIT_SECONDS = 600.0
BUDGET_SECONDS = 60.0
#: Config with a known cap, so the documented "capped" claim is checkable.
CONFIG = {"server": {"max_retry_after_seconds": 3600}}
#: Token evidence emitted on every refusal (the 2026-10-25 incident's numbers).
NEW_TOKENS = 31_246
BUDGET_TOKENS = 25_000


def _contract_section() -> str:
    """The documented per-reason contract section, heading to next heading."""
    text = ROUTING_DOC.read_text()
    start = text.index(SECTION_HEADING)
    rest = text[start + len(SECTION_HEADING) :]
    end = re.search(r"\n#{2,3} ", rest)
    return rest[: end.start()] if end else rest


def _contract_rows() -> dict[str, dict[str, str]]:
    """Parse the documented contract table into ``{reason: {column: cell}}``.

    Column names are backtick-stripped so the assertions can name
    ``Retry-After`` / ``retryable`` / ``retry_at`` directly.
    """
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in _contract_section().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = [cell.strip("`") for cell in cells]
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # markdown separator row
        reason = cells[0].strip("`")
        rows[reason] = dict(zip(header, cells))
    assert header is not None, "the contract section must contain a table"
    return rows


def _documented_timing(reason: str) -> provider.RefusalRetryTiming:
    """Replay a documented row through the production decision."""
    return provider._refusal_retry_timing(
        reason,
        eta_prefill_seconds=PROJECTED_WAIT_SECONDS,
        budget_seconds=BUDGET_SECONDS,
        projected_wait_seconds=PROJECTED_WAIT_SECONDS,
        advertised_retry_after=(COOLDOWN_REMAINDER_SECONDS if reason in REAL_EXPIRY_REASONS else None),
        config=CONFIG,
    )


def test_documented_reasons_are_exactly_the_emitted_vocabulary():
    """One documented row per emitted reason — no invented, no missing reason."""
    documented = set(_contract_rows())

    assert documented == EMITTED_REFUSAL_REASONS, (
        "routing.md's per-reason retry table must document every refusal reason "
        f"and nothing else; missing={sorted(EMITTED_REFUSAL_REASONS - documented)} "
        f"unexpected={sorted(documented - EMITTED_REFUSAL_REASONS)}"
    )


def test_documented_no_timer_rows_match_the_production_constant():
    """The rows claiming \"no Retry-After\" are the production not-retryable set."""
    rows = _contract_rows()
    documented_not_retryable = {
        reason for reason, row in rows.items() if row["retryable"].strip("`").lower() == "false"
    }

    assert documented_not_retryable == set(provider.NOT_TIME_RETRYABLE_REFUSAL_REASONS), (
        "the documented 'compact, do not wait' reasons must equal provider.NOT_TIME_RETRYABLE_REFUSAL_REASONS"
    )
    for reason in documented_not_retryable:
        assert rows[reason]["Retry-After"].lower() == "absent", (
            f"{reason}: a not-time-retryable refusal advertises no Retry-After"
        )
        assert rows[reason]["retry_at"].lower() == "absent", (
            f"{reason}: a not-time-retryable refusal advertises no retry_at"
        )


@pytest.mark.parametrize("reason", sorted(EMITTED_REFUSAL_REASONS))
def test_documented_timer_matches_the_production_decision(reason):
    """Each documented row must describe what the refusal path really does."""
    row = _contract_rows()[reason]
    timing = _documented_timing(reason)

    assert (row["retryable"].strip("`").lower() == "true") is timing.retryable, (
        f"{reason}: documented retryable={row['retryable']!r} but the "
        f"implementation reports retryable={timing.retryable}"
    )
    assert (row["Retry-After"].lower() != "absent") is (timing.retry_after is not None), (
        f"{reason}: documented Retry-After={row['Retry-After']!r} but the "
        f"implementation advertises {timing.retry_after}"
    )
    assert (row["retry_at"].lower() != "absent") is (timing.retry_at is not None)


@pytest.mark.parametrize("reason", sorted(WAIT_BUDGET_REASONS))
def test_documented_wait_budget_rows_name_the_overshoot_and_the_cap(reason):
    """ETA rows say the number is wait-minus-budget, floored and capped."""
    cell = _contract_rows()[reason]["Retry-After"].lower()

    assert "budget" in cell, f"{reason}: the documented timer is the overshoot over the wait budget"
    assert "capped" in cell, f"{reason}: the documented timer is bounded by max_retry_after_seconds"


@pytest.mark.parametrize("reason", sorted(REAL_EXPIRY_REASONS))
def test_documented_real_expiry_rows_are_not_eta_formula_rows(reason):
    """A real-expiry row is described by its expiry, not by wait-minus-budget."""
    cell = _contract_rows()[reason]["Retry-After"].lower()

    assert "budget" not in cell, (
        f"{reason}: a non-admission skip's timer comes from real expiry state, not from the ETA formula"
    )
    assert "cooldown" in cell or "expiry" in cell or "lease" in cell, (
        f"{reason}: the documented timer must name its real expiry source"
    )


def test_every_emitted_refusal_header_is_documented():
    """Doc drift guard: a newly emitted refusal header must be documented."""
    retryable = _documented_timing("eta_exceeds_budget")
    not_retryable = _documented_timing("prefill_budget_exceeded")
    section = _contract_section()

    emitted = set(provider._refusal_timing_headers(retryable, NEW_TOKENS, BUDGET_TOKENS)) | set(
        provider._refusal_timing_headers(not_retryable, NEW_TOKENS, BUDGET_TOKENS)
    )

    assert emitted, "the refusal path must emit header evidence"
    for header in sorted(emitted):
        assert header in section, (
            f"refusal header {header!r} is emitted but not documented in routing.md's per-reason retry contract"
        )


def test_documented_retry_at_is_the_same_instant_as_retry_after():
    """``retry_at`` is documented as an absolute RFC 3339 UTC instant."""
    section = _contract_section()
    before = datetime.now(UTC)
    timing = _documented_timing("eta_exceeds_budget")
    after = datetime.now(UTC)

    assert timing.retry_at is not None
    assert "RFC 3339 UTC" in section, "the docs must state retry_at's wire format (RFC 3339 UTC)"
    parsed = datetime.fromisoformat(timing.retry_at)
    assert parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed), (
        f"retry_at must be UTC, got {timing.retry_at!r}"
    )
    assert (parsed - before).total_seconds() >= (timing.retry_after or 0) - 1, (
        f"retry_at was generated no earlier than now + Retry-After: {timing.retry_at!r}"
    )
    assert (parsed - after).total_seconds() <= (timing.retry_after or 0) + 1, (
        f"retry_at was generated no later than now + Retry-After: {timing.retry_at!r}"
    )


def test_local_preferred_is_documented_as_pinned_while_deferrable():
    """AC6: pinned while deferrable; a too-large request is never refused.

    ``local-preferred`` is one intent with two outcomes, so the contract must
    state both halves: a ``deferrable`` request is refused (with a real retry
    timer) instead of being moved to a remote provider, while a ``too_large``
    request is promoted to the remote tiers instead of being refused for local
    capacity. Pinning only the first half would let the promotion rule silently
    disappear from the contract.
    """
    section = _contract_section()
    plain = " ".join(section.lower().split())

    assert "pinned while deferrable" in plain, (
        "the contract must state that a local-preferred request is pinned while "
        "its capacity verdict is deferrable"
    )
    assert "X-Local-Route: refused" in section, (
        "the refusal shape (429 + X-Local-Route: refused) must be documented"
    )
    assert "never refused for local capacity" in plain, (
        "the contract must state that a too-large request is promoted to a "
        "remote tier, never refused for local capacity"
    )
