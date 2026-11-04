"""Doc/code drift pin: the operator doc's reason list == the vocabulary.

LP-0MW1VVOJZ008YUPH requires the `slots_stale_reason` list in
``docs/local-slots-observability.md`` to match
``observability.SLOTS_STALE_REASONS`` exactly, so a vocabulary change cannot
leave the prose stale (mirrors ``test_slots_model_resolution_docs.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "local-slots-observability.md"
TABLE_HEADING = "Every value is a member of"

DOC_REASON_RE = re.compile(
    r"\| (`[a-z_]+`) \|",  # table cells in the reason/action table
)


def _doc_reasons() -> set[str]:
    text = DOC.read_text(encoding="utf-8")
    table = text.split(TABLE_HEADING, 1)[0]
    start = table.find("| `router_model_missing` |")
    if start < 0:
        return set()
    rows = table[start:]
    return {m.group(1).strip("`") for m in DOC_REASON_RE.finditer(rows)}


class TestDocVocabularyDrift:
    def test_doc_exists(self):
        assert DOC.exists(), "docs/local-slots-observability.md is missing"

    def test_doc_carries_the_reason_table(self):
        text = DOC.read_text(encoding="utf-8")
        assert "| `router_model_missing` |" in text
        assert "Operator action" in text

    def test_doc_list_matches_code_vocabulary_exactly(self):
        from proxy.observability import SLOTS_STALE_REASONS

        documented = _doc_reasons()
        assert documented, "could not extract the reason table from the doc"
        assert documented == set(SLOTS_STALE_REASONS), (
            "docs/local-slots-observability.md lists "
            f"{sorted(documented - set(SLOTS_STALE_REASONS))} extra and is "
            f"missing {sorted(set(SLOTS_STALE_REASONS) - documented)} "
            "relative to observability.SLOTS_STALE_REASONS"
        )

    def test_doc_states_recovery_needs_no_restart(self):
        text = DOC.read_text(encoding="utf-8")
        assert "no restart" in text.lower()
