"""Rendered per-metric detail block (LP-0MW1BUAEG006FMHX).

The integration contract in ``test_performance_metrics_report.py`` pins the
block's position, its one-subsection-per-metric set, the four labelled
paragraphs and the id-first heading shape. This module pins what that contract
does not:

- the rendered prose is the committed defaults' copy, so the renderer supplies
  no metric prose of its own and the YAML stays the single source of truth;
- each subsection links to the data section its figures live in instead of
  repeating that section's tables;
- a metric whose retained best three measurements span configuration eras is
  flagged inside its own subsection, and a like-era or unknown-era history is
  not flagged.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import performance_metrics as pm  # noqa: E402
import reporting  # noqa: E402

from tests import fixtures  # noqa: E402

DETAILS_HEADING = "## Performance metric details"
WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)
FINGERPRINT = "slots=2,ctx=131072"
ERA_FLAG = "**Configuration eras:**"
MISSING_COPY = "not documented"


def _summary() -> aggregation.AnalysisResult:
    events = [
        event
        for event in (log_parser.parse_log_line(line) for line in fixtures.E2E_LINES)
        if event is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _report(
    *,
    metrics_state: Path | None = None,
    metrics_config: Path | None = None,
) -> str:
    return reporting.build_report(
        _summary(),
        None,
        metrics_state=metrics_state,
        metrics_config=metrics_config,
    )


def _detail_blocks(report: str) -> dict[str, str]:
    """The rendered body of each ``### <metric id> — <label>`` subsection."""
    section = report.split(DETAILS_HEADING, 1)[1].split("\n## ", 1)[0]
    blocks: dict[str, str] = {}
    for block in re.split(r"^### ", section, flags=re.M)[1:]:
        heading, _, body = block.partition("\n")
        blocks[heading.split()[0]] = body
    return blocks


def _seed(state_path: Path, metric_id: str, entries: list[tuple[str, float, str]]) -> None:
    """Write a live state file holding *entries* as (window, value, fingerprint)."""
    state = pm.new_state()
    for window, value, fingerprint in entries:
        pm.record_measurement(
            state,
            metric_id,
            window=window,
            value=value,
            date=window,
            fingerprint=fingerprint,
        )
    pm.save_state(state, state_path)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _slug(heading: str) -> str:
    """GitHub-style in-page anchor for a ``## Title`` heading."""
    return "#" + re.sub(r"[^a-z0-9 -]", "", heading.lstrip("#").strip().lower()).replace(
        " ", "-"
    )


# ---------------------------------------------------------------------------
# Committed defaults are the single source of truth
# ---------------------------------------------------------------------------


class TestDetailCopy:
    def test_each_subsection_renders_the_committed_copy_verbatim(self):
        defaults = pm.load_defaults()
        blocks = _detail_blocks(_report())

        for metric in pm.METRIC_REGISTRY:
            copy = pm.detail_copy(metric.id, defaults)
            body = _norm(blocks[metric.id])
            for field in ("why", "calculation", "levers", "trade_offs"):
                assert _norm(copy[field]) in body, f"{metric.id}: {field} not rendered"

    def test_a_metric_without_copy_is_reported_as_undocumented(self, tmp_path):
        # A partial custom defaults file must not crash the report and must not
        # tempt the renderer into inventing prose for the missing fields.
        config = tmp_path / "metrics.yaml"
        config.write_text(
            "metrics:\n  gpu_busy:\n    why: \"Only an importance line.\"\n",
            encoding="utf-8",
        )

        body = _detail_blocks(_report(metrics_config=config))["gpu_busy"]

        assert "Only an importance line." in body
        assert MISSING_COPY in body.lower()


# ---------------------------------------------------------------------------
# Links to the existing data sections, no duplicated tables
# ---------------------------------------------------------------------------


class TestDataSectionLinks:
    def test_each_subsection_links_to_the_section_it_derives_from(self):
        report = _report()
        blocks = _detail_blocks(report)
        heading_slugs = {
            _slug(line) for line in report.splitlines() if line.startswith("## ")
        }

        for metric in pm.METRIC_REGISTRY:
            anchor = pm.detail_copy(metric.id)["data_section_anchor"]
            target = _slug(anchor)
            assert f"]({target})" in blocks[metric.id], (
                f"{metric.id}: no link to its data section {anchor!r}"
            )
            # Only assert resolution where the data section is rendered: some
            # data sections (speed stats, GPU telemetry) need inputs this
            # fixture window has none of.
            if anchor in report:
                assert target in heading_slugs, (
                    f"{metric.id}: {anchor!r} link does not resolve to the section"
                )

    def test_subsections_repeat_no_section_tables(self):
        for metric_id, body in _detail_blocks(_report()).items():
            rows = [line for line in body.splitlines() if line.startswith("|")]
            assert not rows, f"{metric_id}: detail section duplicates table rows"


# ---------------------------------------------------------------------------
# Cross-configuration (era) flag
# ---------------------------------------------------------------------------


class TestCrossEraFlag:
    def test_flags_the_metric_whose_retained_history_spans_eras(self, tmp_path):
        state_path = tmp_path / "state.yaml"
        other = pm.configuration_fingerprint(1, 262144)
        _seed(
            state_path,
            "decode_speed",
            [
                ("2026-07-01", 41.2, FINGERPRINT),
                ("2026-07-02", 42.0, FINGERPRINT),
                ("2026-07-03", 39.0, other),
            ],
        )

        blocks = _detail_blocks(_report(metrics_state=state_path))

        flagged = blocks["decode_speed"].split(ERA_FLAG, 1)[1]
        assert FINGERPRINT in flagged
        assert other in flagged
        for metric_id, body in blocks.items():
            if metric_id != "decode_speed":
                assert ERA_FLAG not in body, f"{metric_id}: false cross-era flag"

    def test_no_flag_for_one_era_unknown_era_or_no_history(self, tmp_path):
        same_era = tmp_path / "same_era.yaml"
        _seed(
            same_era,
            "decode_speed",
            [
                ("2026-07-01", 41.2, FINGERPRINT),
                ("2026-07-02", 42.0, FINGERPRINT),
                ("2026-07-03", 39.0, FINGERPRINT),
            ],
        )
        backfilled = tmp_path / "backfilled.yaml"
        _seed(
            backfilled,
            "decode_speed",
            [
                ("2026-07-01", 41.2, pm.UNKNOWN_FINGERPRINT),
                ("2026-07-02", 42.0, pm.UNKNOWN_FINGERPRINT),
                ("2026-07-03", 39.0, pm.UNKNOWN_FINGERPRINT),
            ],
        )

        for state_path in (same_era, backfilled, tmp_path / "absent.yaml"):
            body = _detail_blocks(_report(metrics_state=state_path))["decode_speed"]
            assert ERA_FLAG not in body, state_path.name
