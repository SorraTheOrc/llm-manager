"""No recommendation may advise raising the local context size.

The local context is bounded by the model (Qwen3, 262144 native) and the Strix
Halo unified-memory budget, so "raise the ctx-size" is advice the operator
cannot act on. Before LP-0MW2BD6G8001E960 four rules repeated it nightly and
two of them cited a work item that had already completed.

This module pins the rule in two places so it cannot silently return:

- **Generated advice** — every recommendation produced from a result that
  fires all four large-context rules, scanned for the forbidden phrases in
  ``recommendations.FORBIDDEN_CTX_ADVICE``.
- **Rendered documentation** — ``SKILL.md``, ``REFERENCE.md``, ``README.md``
  and the committed metric defaults, which carried the same advice in prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import recommendations  # noqa: E402

SKILL_ROOT = Path(__file__).resolve().parents[1]
DOCS = ("SKILL.md", "REFERENCE.md", "README.md")
METRICS_DEFAULTS = SKILL_ROOT / "metrics" / "performance_metrics.yaml"

#: The completed work item the old advice used to cite. Recommending a finished
#: item as an open action is its own defect, so it is banned alongside the
#: ctx-size phrasing.
COMPLETED_CTX_WORK_ITEM = "LP-0MSAOQTJS000FFVM"


class TestTheGuardItself:
    """A guard that cannot detect its target is worse than no guard."""

    def test_every_forbidden_phrase_is_detected(self):
        for phrase in recommendations.FORBIDDEN_CTX_ADVICE:
            assert recommendations.ctx_advice_in(f"Please {phrase} now.") == phrase

    def test_detection_is_case_insensitive(self):
        assert (
            recommendations.ctx_advice_in("RAISE THE LOCAL CTX-SIZE")
            == "raise the local ctx-size"
        )

    def test_ordinary_copy_is_not_flagged(self):
        assert (
            recommendations.ctx_advice_in(
                "The context ceiling is fixed by the model and the hardware available."
            )
            is None
        )


class TestRenderedDocumentation:
    def test_no_document_advises_raising_the_context(self):
        for name in DOCS:
            text = (SKILL_ROOT / name).read_text(encoding="utf-8")
            offending = recommendations.ctx_advice_in(text)
            assert offending is None, f"{name} still advises {offending!r}"

    def test_metric_defaults_name_the_fixed_ceiling(self):
        text = METRICS_DEFAULTS.read_text(encoding="utf-8")
        offending = recommendations.ctx_advice_in(text)
        assert offending is None, f"performance_metrics.yaml still advises {offending!r}"
        assert "fixed by the model and the hardware" in text

    def test_the_completed_work_item_is_not_cited(self):
        for name in DOCS:
            text = (SKILL_ROOT / name).read_text(encoding="utf-8")
            assert COMPLETED_CTX_WORK_ITEM not in text, (
                f"{name} still cites the completed {COMPLETED_CTX_WORK_ITEM}"
            )
