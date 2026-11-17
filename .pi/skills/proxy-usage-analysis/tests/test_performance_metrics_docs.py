"""Documentation contract for the performance-metrics scorecard.

LP-0MW1BUBDS001Q9F6 documents the scorecard in `SKILL.md`, `REFERENCE.md` and
`README.md`. These tests keep that documentation tied to the shipped
implementation rather than to a snapshot of its prose:

- the metric definitions are driven by ``METRIC_REGISTRY``, so adding,
  renaming or re-pointing a metric without documenting it fails here;
- the file schemas and the backfill command are asserted by their stable
  identifiers (file names, keys, flag names), not by wording;
- the unchanged 05:00 cron invocation is asserted on the command line itself,
  which is the only claim in the docs a reader could act on wrongly.

Assertions are deliberately about facts an operator needs (which metric, which
unit, which direction, which file, which flag) so the docs may be rewritten
freely without touching this module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import performance_metrics  # noqa: E402

SKILL_DOC = SKILL_ROOT / "SKILL.md"
REFERENCE_DOC = SKILL_ROOT / "REFERENCE.md"
README_DOC = SKILL_ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reference_documents_every_registered_metric_definition():
    """Each registry metric is documented with its unit, direction and source."""
    reference = _read(REFERENCE_DOC)

    for metric in performance_metrics.METRIC_REGISTRY:
        row = next(
            (
                line
                for line in reference.splitlines()
                if line.startswith(f"| `{metric.id}` |")
            ),
            None,
        )
        assert row is not None, f"REFERENCE.md has no definition row for {metric.id}"
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        _, label, unit, direction, source = cells
        assert label == metric.label
        assert unit == metric.unit
        expected_direction = (
            "higher is better" if metric.higher_is_better else "lower is better"
        )
        assert direction == expected_direction
        # Prose cells may wrap log tokens in backticks; compare the words.
        assert source.replace("`", "") == metric.source.replace("`", "")


def test_reference_documents_both_file_schemas_and_the_measurement_fields():
    """The committed defaults and the live state file are both specified."""
    reference = _read(REFERENCE_DOC)

    assert "metrics/performance_metrics.yaml" in reference
    assert performance_metrics.STATE_FILENAME in reference
    # Detail-copy fields the committed defaults may declare.
    for field in performance_metrics.DETAIL_FIELDS:
        assert field in reference, f"detail-copy field {field} is undocumented"
    # Measurement fields the live state file stores.
    for field in ("window", "value", "date", "fingerprint"):
        assert field in reference, f"measurement field {field} is undocumented"
    assert "read-only at runtime" in reference.lower()


def test_reference_documents_the_backfill_command_and_its_example_run():
    reference = _read(REFERENCE_DOC)

    assert "--seed-metrics-history" in reference
    assert "Seeded " in reference and "already present" in reference
    assert "Idempotent per window" in reference
    # Metrics the archive cannot supply must be named as skipped, not guessed.
    assert "not rendered in the archived report" in reference


def test_skill_documents_the_scorecard_surfaces_and_four_flags():
    skill = _read(SKILL_DOC)

    assert "## Performance metrics summary" in skill
    assert "## Performance metric details" in skill
    for column in ("Metric", "Baseline", "Target", "Tolerance", "Actual", "Result"):
        assert column in skill, f"scorecard column {column} is undocumented"
    for flag in (
        "--metrics-config",
        "--metrics-state",
        "--no-metrics-update",
        "--seed-metrics-history",
    ):
        assert flag in skill, f"{flag} is undocumented in SKILL.md"


def test_skill_documents_the_tolerance_column_and_its_default():
    """LP-0MWADHS83006QFQS AC3/AC7: the scorecard documents the visible band,
    the 10 % default and the per-metric override."""
    skill = _read(SKILL_DOC)

    assert "| Tolerance |" in skill
    assert "±10 %" in skill
    assert "within the tolerance band of the target" in skill
    assert "tolerance_pct" in skill


def test_reference_documents_the_tolerance_rule_and_override_field():
    """AC2/AC5/AC7: the verdict rule, the defaults-file field and the
    zero-width band that follows from a zero target."""
    reference = _read(REFERENCE_DOC)

    assert "tolerance_pct" in reference
    assert "±10 %" in reference
    assert "zero-width band" in reference.lower()


def test_docs_describe_the_strict_zero_band_as_deliberate():
    """LP-0MWJOUY4E002FNHP: an explicit zero band is documented strictness,
    never read as an absent value that would restore the 10 % default."""
    skill = _read(SKILL_DOC)
    reference = _read(REFERENCE_DOC)

    for name, doc in (("SKILL.md", skill), ("REFERENCE.md", reference)):
        assert "tolerance_pct: 0" in doc, f"{name} omits the strict zero band"
        assert "±0 %" in doc, f"{name} omits the rendered zero band"
    assert "fallback_rate" in reference


def test_docs_describe_the_three_redefined_metric_semantics():
    """LP-0MWJOV2S7001CVS5: the three redefined scorecard rows are documented
    as they are now calculated — a strict Fallback-rate verdict, the 80 %
    context-pressure measure and the raw prefill-ETA error."""
    skill = _read(SKILL_DOC)
    reference = _read(REFERENCE_DOC)

    for name, doc in (("SKILL.md", skill), ("REFERENCE.md", reference)):
        assert "context_pressure" in doc, f"{name} omits context_pressure"
        assert "80 %" in doc, f"{name} omits the 80 % context-pressure threshold"
        assert "prefill_eta_accuracy" in doc, f"{name} omits prefill_eta_accuracy"
        assert "median absolute relative error" in doc, (
            f"{name} omits the raw prefill-ETA error measure"
        )
        assert "lower" in doc.lower()
        # The strict Fallback-rate verdict: a zero, one-sided band.
        assert "tolerance_pct: 0" in doc, f"{name} omits the strict fallback band"
        # The retired accuracy conversion must not be the documented
        # calculation: the docs state the raw error, unclamped, so an error
        # above 100 % reads above 100 %.
        assert "above 100 %" in doc, f"{name} omits the unbounded-error guarantee"


def _section(text: str, heading: str) -> str:
    """Body of the ``heading`` section (up to the next heading of any level)."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == heading),
        None,
    )
    assert start is not None, f"heading {heading!r} is missing"
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def test_reference_documents_the_definition_change_history_reset():
    """A metric whose definition changed has its history reset with a recorded
    reason, so baseline/target are never computed across two definitions."""
    reference = _read(REFERENCE_DOC)

    assert "reset_metric_history" in reference
    section = _section(reference, "### Definition-change history resets")
    assert "definition_change" in section
    assert "notes" in section
    # Both redefined metrics are named as reset, with their new semantics.
    assert "context_pressure" in section
    assert "80 %" in section
    assert "prefill_eta_accuracy" in section
    assert "median absolute relative error" in section
    assert "lower" in section.lower()
    # fallback_rate history is deliberately retained (only its band moved).
    assert "fallback_rate" in section
    assert "retain" in section.lower()
    # A reset reads as Insufficient history, never as a fabricated target.
    assert "Insufficient history" in section


def test_live_state_schema_documents_the_notes_block():
    """The state-file `notes` block is part of the documented schema."""
    reference = _read(REFERENCE_DOC)

    assert "notes:" in reference
    assert "event: definition_change" in reference


def test_readme_describes_the_redefined_metric_semantics():
    """The skill README names the same three semantics as the other surfaces."""
    readme = _read(README_DOC)

    assert "context_pressure" in readme or "80 %" in readme
    assert "median absolute relative error" in readme


def test_skill_states_the_0500_cron_invocation_is_unchanged():
    """The documented cron line still carries no performance-metric flag."""
    skill = _read(SKILL_DOC)

    cron_lines = [line for line in skill.splitlines() if line.startswith("0 5 * * *")]
    assert cron_lines, "SKILL.md no longer documents the 05:00 cron line"
    for line in cron_lines:
        assert "analyze_proxy_usage.py" in line
        assert "--metrics" not in line, "the cron line must need no metrics flag"
    assert re.search(r"cron invocation is unchanged", skill)


def test_readme_pipeline_overview_names_the_scorecard_state_and_backfill():
    readme = _read(README_DOC)

    for token in ("scorecard", performance_metrics.STATE_FILENAME, "--seed-metrics-history"):
        assert token in readme, f"README pipeline overview omits {token}"
