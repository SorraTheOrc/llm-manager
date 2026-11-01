"""Docs contract: the free-slot (direct) path's refusal-reason scoping.

LP-0MVT3TBQW004W5LM (parent LP-0MVRO3AHN002UM1Q AC1/AC6/AC7). The deliverable
is prose in ``proxy/docs/routing.md``, so the assertions pin the *rule* the
prose must state rather than a copy of the table:

- the "Local-only is enforced on every exit" exit table's direct free-slot row
  must scope ``eta_exceeds_budget_direct`` to the free-slot **ETA/budget** bound
  and name the reasons that keep their own governing token instead;
- the per-reason retry contract must state the same scoping, because its table
  is reason-keyed — a reader must be told the free-slot label belongs to the
  ETA/budget bound alone and that the other rows apply unchanged on that path.

The production behaviour these statements describe is pinned end-to-end by
``test_local_admission_guard.py::test_direct_path_names_the_governing_reason_end_to_end``
and ``::test_direct_free_slot_budget_refusal_keeps_its_reason``, and at the
guard level by ``test_prefill_eta_deadline.py``'s direct-path label tests, so a
code change that widens or narrows the label set fails there. This module
fails when the docs stop saying it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"

#: The operator-facing default configuration whose guard comment documents
#: the same label mapping (LP-0MVXZA6L30000QYD — the parent audit's AC6 gap:
#: the config stated the direct decision *always* logs the ETA/budget token).
CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"

#: The exit-table row that documents the free-slot dispatch path.
DIRECT_PATH_EXIT = "Admission guard, direct free-slot dispatch path"

#: The free-slot ETA/budget label — the *only* token the direct path adds.
DIRECT_PATH_TOKEN = "eta_exceeds_budget_direct"

#: Reasons the free-slot guard keeps verbatim instead of relabelling them
#: (``provider.py``'s free-slot branch and ``_prefill_admission_guard``'s
#: ``log_label`` rule).
GOVERNING_REASONS = {
    "prefill_eta_exceeds_deadline",
    "prefill_eta_exceeds_safety_budget",
    "prefill_budget_exceeded",
}

#: Heading of the per-reason retry contract; the section ends at the next
#: level-3 heading.
RETRY_SECTION_HEADING = "### Per-reason retry contract (LP-0MVNTT4X8009IO2X)"


def _direct_path_row() -> str:
    """The single exit-table row documenting the free-slot dispatch path."""
    rows = [
        line
        for line in ROUTING_DOC.read_text().splitlines()
        if line.startswith(f"| {DIRECT_PATH_EXIT} ")
    ]
    assert len(rows) == 1, (
        "routing.md must carry exactly one exit row for the direct free-slot "
        f"path; got {rows!r}"
    )
    return rows[0]


def _ticked_spans(text: str) -> set[str]:
    return set(re.findall(r"`([^`]+)`", text))


def _retry_scoping_paragraph() -> str:
    """The retry-contract paragraph that scopes the free-slot label."""
    text = ROUTING_DOC.read_text()
    start = text.index(RETRY_SECTION_HEADING)
    end = text.find("\n### ", start + len(RETRY_SECTION_HEADING))
    section = text[start : end if end != -1 else len(text)]
    paragraphs = [
        " ".join(block.split())
        for block in re.split(r"\n\s*\n", section)
        if DIRECT_PATH_TOKEN in block and "free-slot" in block
    ]
    assert len(paragraphs) == 1, (
        "the per-reason retry contract must carry exactly one paragraph that "
        f"scopes {DIRECT_PATH_TOKEN!r} to the free-slot path; got {paragraphs!r}"
    )
    return paragraphs[0]


def test_direct_path_row_scopes_the_token_to_the_eta_budget_bound():
    """AC1/AC6: only the free-slot ETA/budget bound emits the direct token.

    The row's own token is the ETA/budget label; the prose must say *only* and
    name every reason that keeps its own token instead. Without that scoping an
    operator reads the row as "the direct path always reports this token" —
    exactly the mislabelling the sibling tests exist to prevent.
    """
    row = _direct_path_row()
    # Cell 2 is the token cell (cell 1 is the exit description).
    token_cell = " ".join(row.strip().strip("|").split("|")[1].split())

    assert token_cell.startswith(f"`{DIRECT_PATH_TOKEN}`"), (
        "the direct-path row must pin the ETA/budget token first; got "
        f"{token_cell!r}"
    )
    assert "only" in token_cell.lower(), (
        "the direct-path row must state that ONLY the free-slot ETA/budget "
        f"bound emits {DIRECT_PATH_TOKEN!r}; got {token_cell!r}"
    )
    missing = GOVERNING_REASONS - _ticked_spans(token_cell)
    assert not missing, (
        "the direct-path row must name every reason that keeps its own "
        f"governing token; missing={sorted(missing)}"
    )


def test_retry_contract_states_the_path_never_changes_the_reason():
    """AC1/AC6: the reason-keyed retry table carries the same scoping rule.

    The table lists one row per reason, so a reader must be told that
    ``eta_exceeds_budget_direct`` is the free-slot label of the ETA/budget
    bound alone and that the deadline, safety-budget and cold-prefill rows
    apply unchanged wherever they fire — otherwise the direct path looks like
    it has its own timing contract.
    """
    prose = _retry_scoping_paragraph()

    assert "only" in prose.lower(), (
        "the retry contract must scope the free-slot label to the ETA/budget "
        f"bound; got {prose!r}"
    )
    missing = GOVERNING_REASONS - _ticked_spans(prose)
    assert not missing, (
        "the retry contract must name the reasons that keep their own token on "
        f"the free-slot path; missing={sorted(missing)}"
    )
    assert "both" in prose.lower() and "path" in prose.lower(), (
        "the retry contract must state that the reason is the same on both "
        f"dispatch paths; got {prose!r}"
    )


# ---------------------------------------------------------------------------
# The shipped default configuration (LP-0MVXZA6L30000QYD / parent AC6): the
# config comment that documents the direct decision must state the same
# scoping as routing.md, or an operator reading config.yaml is misled into
# the 2026-10-25 retry-storm shape (a timer advertised for a refusal that
# waiting cannot heal). Textual check — the paragraph lives inside a YAML
# comment block, so the assertion pins the *rule* the prose must state,
# matching the routing.md helpers above.
# ---------------------------------------------------------------------------


def _config_direct_decision_paragraph() -> str:
    """The config comment paragraph naming the direct decision, or fail."""
    lines = CONFIG_YAML.read_text().splitlines()
    start = [i for i, ln in enumerate(lines) if "The direct decision logs" in ln]
    assert len(start) == 1, (
        "config.yaml must describe the direct decision exactly once; got "
        "occurrences at {start!r}"
    )
    para = [lines[start[0]]]
    for ln in lines[start[0] + 1:]:
        stripped = ln.strip()
        if not stripped or stripped == "#":
            break
        para.append(ln)
    return " ".join(para)


def test_config_yaml_scopes_the_direct_token_to_the_eta_budget_bound():
    """AC1: config.yaml must not contradict routing.md's scoping rule.

    The parent audit (LP-0MVRO3AHN002UM1Q AC6) found the shipped config said
    the direct decision logs ``eta_exceeds_budget_direct`` unconditionally.
    The comment must now state that only the free-slot ETA/budget bound keeps
    that token and name every reason that keeps its own instead.
    """
    para = _config_direct_decision_paragraph()
    assert "the direct decision logs" in para.lower()
    assert "only" in para.lower(), (
        "config.yaml must scope eta_exceeds_budget_direct to the free-slot "
        f"ETA/budget bound; got {para!r}"
    )
    missing = GOVERNING_REASONS - _ticked_spans(para)
    assert not missing, (
        "config.yaml must name every reason that keeps its own governing "
        f"token on the direct path; missing={sorted(missing)}"
    )
    assert "free_slot=true" in para, (
        "config.yaml's direct-decision comment must keep the free_slot marker"
    )
