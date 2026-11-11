"""Documentation contract for the turn-first fallback attribution rules.

LP-0MW8U3B5R005KMO1 documents, in the places an operator or a future
maintainer actually reads, how a session's fallback reason is attributed:

- the turn-first join (an exact ``turn=`` match on the session's own evidence
  wins over the timestamp) and *why* it has to win — the proxy emits
  ``Fallback triggered`` after the pre-flight has finished the remote stream it
  describes, so a timestamp-first filter drops the session's own line;
- the bounded legacy path (session-less lines within
  ``FALLBACK_ATTRIBUTION_WINDOW_SECONDS``) and the timestamp rule that
  turn-less rotated logs keep;
- the move-time clamp, so a late matched line never shifts
  ``context_at_fallback``;
- that ``unattributed`` means *no own evidence*, never "the line arrived late".

The assertions are about the facts a reader needs — which picker decides, which
window bounds the legacy path, which attribution states exist, and which rule
applies when a turn id is missing — not about a snapshot of the prose, so the
docs may be reworded without touching this module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402

SKILL_DOC = SKILL_ROOT / "SKILL.md"
REFERENCE_DOC = SKILL_ROOT / "REFERENCE.md"
README_DOC = SKILL_ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract(text: str, start: str, end: str | None = None) -> str:
    """Return the slice of *text* starting at *start*, optionally up to *end*."""
    index = text.index(start)
    chunk = text[index:]
    if end is not None:
        stop = chunk.find(end, len(start))
        if stop != -1:
            chunk = chunk[:stop]
    return chunk


def test_module_block_documents_the_turn_first_order_and_the_clamp():
    """The module-level attribution block states how the move is attributed."""
    doc = aggregation.__doc__ or ""

    # The shared picker is the one place the order lives, and the module block
    # has to say so — otherwise a reader assumes the timestamp filter decides.
    assert "_pick_turn_first_evidence" in doc
    assert "turn-first" in doc
    assert "turn=" in doc
    # The proxy logs the fallback line after the stream it describes.
    assert "after" in doc and "Fallback triggered" in doc
    # The legacy path is bounded, and the recorded move time is clamped.
    assert "FALLBACK_ATTRIBUTION_WINDOW_SECONDS" in doc
    assert "clamp" in doc.lower()


def test_attribute_fallback_docstring_documents_order_legacy_path_and_clamp():
    """``_attribute_fallback()``'s docstring carries the same three facts."""
    doc = aggregation._attribute_fallback.__doc__ or ""

    assert "turn-first" in doc
    assert "turn=" in doc
    # The bounded legacy path, by constant name (stable identifier).
    assert "FALLBACK_ATTRIBUTION_WINDOW_SECONDS" in doc
    # The clamp keeps context_at_fallback anchored at/before the first stream.
    assert "clamp" in doc.lower()
    assert "context_at_fallback" in doc


def test_picker_docstring_names_every_own_evidence_tier_it_serves():
    """The shared picker documents the tiers that rely on it."""
    doc = aggregation._pick_turn_first_evidence.__doc__ or ""

    for tier in (
        "routing_skip_local",
        "routing_skip_provider_context",
        "Fallback triggered",
    ):
        assert tier in doc, f"picker docstring does not name the {tier} tier"
    assert "turn=" in doc
    assert "at/before" in doc


def test_reference_limitations_state_the_turn_first_join_and_legacy_rule():
    reference = _read(REFERENCE_DOC)
    bullet = _extract(
        reference,
        "- **Per-session fallback attribution",
        "\n- **Per-turn attribution is exact",
    )

    assert "turn-first" in bullet
    assert "turn=" in bullet
    # The observed cause of the late line: the proxy logs it after the stream.
    assert "Fallback triggered" in bullet
    assert "pre-flight" in bullet and "after" in bullet
    # Turn-less rotated logs keep the timestamp rule.
    assert "at/before" in bullet
    # And the bounded legacy path still applies to session-less lines.
    assert "session-less" in bullet and "60s" in bullet
    # The clamp: the move time never lands after the first remote stream.
    assert "never later" in bullet
    # `unattributed` is about absent own evidence, not a late arrival.
    assert "`unattributed`" in bullet
    assert "no own evidence" in bullet


def test_reference_exact_attribution_claims_match_the_picker():
    """The `turn=` contract paragraph must not over-claim what is turn-joined."""
    reference = _read(REFERENCE_DOC)
    paragraph = _extract(
        reference,
        "- **Exact attribution.**",
        "\n   Local `dispatch_first_byte_ms=`",
    )

    # The join is made by `_attribute_fallback()` for the fallback/skip tiers.
    assert "_attribute_fallback()" in paragraph
    assert "turn" in paragraph
    # Denials are attributed by a timestamp block, not by `turn=` (see the
    # Limitations section), so the paragraph must not claim denial joins.
    assert "denial" not in paragraph.lower()
    # Turn-less lines keep the historical rule, stated in the same paragraph.
    assert "without" in paragraph and "ids" in paragraph


def test_reference_documents_every_attribution_state():
    """Each attribution state the code can report is documented in REFERENCE."""
    reference = _read(REFERENCE_DOC)

    for state in aggregation.FALLBACK_ATTRIBUTION_ORDER:
        assert f"`{state}`" in reference, f"attribution state {state} undocumented"


def test_skill_local_turn_coverage_explains_unattributed_is_no_own_evidence():
    skill = _read(SKILL_DOC)
    section = _extract(
        skill,
        "## Local-turn coverage",
        "## Context size at fallback by percentile band",
    )

    assert "`unattributed`" in section
    assert "own evidence" in section
    # The misreading this rule exists to prevent: a late line is not a miss.
    assert "late" in section
    # A reader must be able to find the exact rule that decides instead.
    assert re.search(r"turn-first|`turn=`", section), (
        "SKILL.md does not point at the turn-first join"
    )


def test_readme_keeps_restart_markers_and_fallback_attribution_apart():
    """README's attribution section is about restart/mode markers, and says so."""
    readme = _read(README_DOC)
    section = _extract(readme, "## Attribution Markers", "## See Also")

    # Fallback attribution is not restated here; it is pointed at REFERENCE.md.
    assert "fallback attribution" in section.lower()
    assert "REFERENCE.md" in section
    # The markers stay documented where they belong.
    assert "source=" in section and "actor=" in section and "reason=" in section
