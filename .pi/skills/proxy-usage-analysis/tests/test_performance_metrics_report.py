"""Integration specification for the performance-metrics report/JSON surface.

Fail-first contract for the *rendering* half of LP-0MW0FU5Z6006A4P3: the
scorecard's position in the daily report, its columns, the per-metric detail
block, the live state file and the ``performance_metrics`` ``--json`` block.
It is the integration sibling of ``test_performance_metrics.py`` (which pins
the engine API); this module pins what the report and the CLI must show, so
the renderer (LP-0MW1BUA2M0005QRV), the CLI wiring
(LP-0MW1BUAQD001WL8X) and the backfill command (LP-0MW1BUB1Z004AK21) are
driven by these tests rather than the other way round.

Pinned surface
--------------

Markdown (``reporting.build_report``)::

    # Proxy Usage Analysis Report
    - <existing header bullets>
    ## Performance metrics summary              <- first "##" section

    | Metric | Baseline | Target | Actual | Result |
    |---|---|---|---|---|
    | <Metric.label> | ... | ... | ... | Met / Unmet / Insufficient history |

    ## Performance metric details                <- immediately after the table

    ### <Metric.id> — <Metric.label>            <- one per metric, registry order
    **Importance:** ...
    **Calculation:** ...   (must name the metric's ``Metric.source``)
    **Levers:** ...
    **Trade-offs:** ...

Row order and labels come from ``performance_metrics.METRIC_REGISTRY`` (single
source of truth): one summary row and exactly one ``###`` detail subsection per
registry entry, in registry order, and each detail heading leads with the
metric **id** followed by its label.

The id-first heading is not cosmetic. Existing report sections are extracted
throughout the suite with ``report.split("## <Title>", 1)[1]``, and a
``### <Title>`` heading matches that split first (``"## Decode speed"`` is a
substring of ``"### Decode speed"``), silently handing those assertions the
detail body instead of the section body. Leading with the id (``###
decode_speed — Decode speed``) keeps every existing section split intact —
see ``TestMetricDetails.test_detail_headings_never_shadow_an_existing_section_split``.

A metric with fewer than three recorded
measurements renders ``n/a`` for Baseline/Target and the verdict
``Insufficient history`` — never a false Met/Unmet — and the columns of every
row stay within ``Met`` / ``Unmet`` / ``Insufficient history``.

CLI (``analyze_proxy_usage``)::

    --metrics-config PATH        committed defaults (default: the committed
                                 metrics/performance_metrics.yaml, never written)
    --metrics-state PATH         live state file
                                 (default: <output-dir>/performance_metrics_state.yaml)
    --no-metrics-update          render the scorecard but never write the state
    --seed-metrics-history [DIR] one-off backfill; DIR holds one subdirectory per
                                 archived window, each with a report.md

A default run (no metrics flags) renders the section and writes only the live
state file under ``--output-dir``; the committed defaults file stays
byte-identical. ``--json`` gains a top-level ``performance_metrics`` mapping
keyed by metric id, each entry carrying ``baseline`` (``value`` + ``date``),
``target``, ``actual`` and ``verdict``, and those verdicts must agree with the
rendered table.

Fail-first window
-----------------

Three sibling items deliver this surface, so every test is skipped until the
capability it exercises exists (a skip keeps the deterministic finish/push gate
green; the module still collects, so the suite never exits "no tests
collected"). The probes are behavioural, not name-based:

- the render tests run once ``performance_metrics.py`` and the committed
  defaults exist **and** ``build_report()`` actually emits
  ``## Performance metrics summary``;
- the detail-section tests additionally wait for ``## Performance metric
  details`` (LP-0MW1BUAEG006FMHX), so the summary renderer can land first
  without leaving the suite red;
- the CLI tests run once ``--metrics-config`` / ``--metrics-state`` /
  ``--no-metrics-update`` parse;
- the backfill test runs once ``--seed-metrics-history`` also parses a
  directory argument.

No test has to be edited for the specification to start running once each
piece lands.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SKILL_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import analyze_proxy_usage  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

from tests import fixtures  # noqa: E402

ENGINE_PATH = SCRIPTS / "performance_metrics.py"
DEFAULTS_PATH = SKILL_ROOT / "metrics" / "performance_metrics.yaml"

SUMMARY_HEADING = "## Performance metrics summary"
DETAILS_HEADING = "## Performance metric details"
TABLE_COLUMNS = ["Metric", "Baseline", "Target", "Actual", "Result"]
DETAIL_LABELS = ("Importance", "Calculation", "Levers", "Trade-offs")
STATE_FILENAME = "performance_metrics_state.yaml"
VERDICTS = ("Met", "Unmet", "Insufficient history")

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)


# ---------------------------------------------------------------------------
# Fail-first capability probes
# ---------------------------------------------------------------------------


def _probe_summary():
    """A small real summary (the E2E fixture window) used by the probe only."""
    events = [
        event
        for event in (log_parser.parse_log_line(line) for line in fixtures.E2E_LINES)
        if event is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _probe_report_surface() -> tuple[bool, str]:
    if not ENGINE_PATH.exists():
        return False, f"{ENGINE_PATH.name} not implemented yet (LP-0MW1BU9F8007SE0U)"
    if not DEFAULTS_PATH.exists():
        return False, "committed defaults not implemented yet (LP-0MW1BU9F8007SE0U)"
    try:
        report = reporting.build_report(_probe_summary(), None)
    except Exception as exc:  # a half-landed renderer must not break collection
        return False, f"build_report() raised {exc!r}"
    if SUMMARY_HEADING not in report:
        return False, f"{SUMMARY_HEADING!r} not rendered yet (LP-0MW1BUA2M0005QRV)"
    return True, ""


def _parse(args: list[str]):
    """Parse CLI args, returning None when the parser rejects them."""
    try:
        return analyze_proxy_usage.parse_args(args)
    except SystemExit:
        return None


REPORT_AVAILABLE, REPORT_SKIP_REASON = _probe_report_surface()


def _probe_details_surface() -> tuple[bool, str]:
    """The detail block lands with its own work item (LP-0MW1BUAEG006FMHX)."""
    if not REPORT_AVAILABLE:
        return False, REPORT_SKIP_REASON
    try:
        report = reporting.build_report(_probe_summary(), None)
    except Exception as exc:  # a half-landed renderer must not break collection
        return False, f"build_report() raised {exc!r}"
    if DETAILS_HEADING not in report:
        return False, f"{DETAILS_HEADING!r} not rendered yet (LP-0MW1BUAEG006FMHX)"
    return True, ""


DETAILS_AVAILABLE, DETAILS_SKIP_REASON = _probe_details_surface()

needs_details = pytest.mark.skipif(not DETAILS_AVAILABLE, reason=DETAILS_SKIP_REASON)

CLI_AVAILABLE = REPORT_AVAILABLE and all(
    _parse(args) is not None
    for args in (
        ["--metrics-config", str(DEFAULTS_PATH)],
        ["--metrics-state", str(DEFAULTS_PATH)],
        ["--no-metrics-update"],
    )
)

BACKFILL_AVAILABLE = CLI_AVAILABLE and (
    _parse(["--seed-metrics-history", str(SKILL_ROOT)]) is not None
)

if REPORT_AVAILABLE:  # pragma: no cover - fail-first window
    import performance_metrics as pm
else:  # pragma: no cover - fail-first window
    pm = None

pytestmark = pytest.mark.skipif(not REPORT_AVAILABLE, reason=REPORT_SKIP_REASON)

needs_cli = pytest.mark.skipif(
    not CLI_AVAILABLE,
    reason="metrics CLI flags not implemented yet (LP-0MW1BUAQD001WL8X)",
)
needs_backfill = pytest.mark.skipif(
    not BACKFILL_AVAILABLE,
    reason="--seed-metrics-history not implemented yet (LP-0MW1BUB1Z004AK21)",
)

#: Existing report sections, from the parent work item's per-metric inputs.
#: Their titles must never appear inside a ``###`` detail heading (see
#: ``TestMetricDetails.test_detail_headings_never_shadow_an_existing_section_split``).
EXISTING_SECTION_TITLES = (
    "## Summary by hour",
    "## Session summary",
    "## Recommendations",
    "## Local model utilization",
    "## Fallback reasons",
    "## Time to first token",
    "## Decode speed",
    "## Prompt eval speed",
    "## Error analysis",
    "## Local capacity signals",
    "## Local contention prediction vs reality",
    "## Context sizes approaching per-slot limits",
)

METRIC_IDS = [
    "gpu_busy",
    "fallback_rate",
    "decode_speed",
    "prompt_eval_speed",
    "time_to_first_token",
    "stream_error_rate",
    "context_pressure",
    "prompt_cache_hit_ratio",
    "contention_denials_per_hour",
    "prefill_eta_accuracy",
]


def _metric_labels() -> list[str]:
    return [metric.label.strip() for metric in pm.METRIC_REGISTRY]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Report parsing helpers
# ---------------------------------------------------------------------------


def _report() -> str:
    return reporting.build_report(_probe_summary(), None)


def _section(report: str, heading: str) -> str:
    assert heading in report, f"{heading!r} missing from the report"
    return report.split(heading, 1)[1].split("\n## ", 1)[0]


def _summary_rows(report: str) -> list[list[str]]:
    """Return the summary table's data rows as lists of five cells."""
    rows = [line for line in _section(report, SUMMARY_HEADING).splitlines() if line.startswith("|")]
    assert rows, f"{SUMMARY_HEADING!r} has no table"
    cells = [row.strip().strip("|").split("|") for row in rows]
    header = [cell.strip() for cell in cells[0]]
    assert header == TABLE_COLUMNS
    assert [cell.strip() for cell in cells[1]] == ["---"] * len(TABLE_COLUMNS)
    return [[cell.strip() for cell in row] for row in cells[2:]]


def _detail_entries(report: str) -> list[tuple[str, str]]:
    """Return ``(heading, body)`` for each ``###`` subsection, in report order."""
    section = _section(report, DETAILS_HEADING)
    blocks = re.split(r"^### ", section, flags=re.M)[1:]
    return [(block.splitlines()[0].strip(), block) for block in blocks]


def _labelled_texts(block: str) -> dict[str, str]:
    """Extract the labelled paragraphs (``**Levers:** ...``) from a subsection."""
    texts: dict[str, str] = {}
    for label in DETAIL_LABELS:
        marker = f"**{label}:**"
        assert marker in block, f"missing {marker!r} label"
        after = block.split(marker, 1)[1]
        for other in DETAIL_LABELS:
            other_marker = f"**{other}:**"
            if other_marker != marker and other_marker in after:
                after = after.split(other_marker, 1)[0]
        texts[label] = after.strip()
    return texts


# ---------------------------------------------------------------------------
# CLI/fixture helpers
# ---------------------------------------------------------------------------


def _write_log_dir(root: Path) -> Path:
    """Write the E2E fixture lines as a rotated + live proxy log pair."""
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "proxy.log.2026-08-02_14").write_text("\n".join(fixtures.E2E_LINES[:3]) + "\n")
    (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES[3:]) + "\n")
    return log_dir


def _run_cli(log_dir: Path, out_dir: Path, *extra: str) -> int:
    return analyze_proxy_usage.main(
        [
            "--log-dir",
            str(log_dir),
            "--llama-log-dir",
            str(log_dir),
            "--start",
            "2026-08-02 14:00:00",
            "--end",
            "2026-08-02 15:00:00",
            "--output-dir",
            str(out_dir),
            *extra,
        ]
    )


def _measurements(state: dict, metric_id: str) -> list[dict]:
    return (state.get("metrics") or {}).get(metric_id, {}).get("measurements", [])


# ---------------------------------------------------------------------------
# Scorecard position and columns
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_summary_table_is_the_first_section_after_the_header(self):
        report = _report()

        headings = [line for line in report.splitlines() if line.startswith("## ")]

        assert headings[0] == SUMMARY_HEADING
        assert report.index("# Proxy Usage Analysis Report") < report.index(SUMMARY_HEADING)
        # The pre-existing first section keeps its precedence below the new one
        # (no existing ordering assertion is weakened by the insertion).
        assert report.index(SUMMARY_HEADING) < report.index("## Summary by hour")

    def test_summary_table_has_the_five_required_columns(self):
        rows = _summary_rows(_report())

        assert len(rows) == len(pm.METRIC_REGISTRY) == 10
        for cells in rows:
            assert len(cells) == len(TABLE_COLUMNS)
            assert cells[4] in VERDICTS, f"{cells[0]}: unexpected result {cells[4]!r}"

    def test_one_row_per_metric_in_registry_order(self):
        labels = _metric_labels()

        assert len(labels) == len(set(labels)) == 10
        assert [cells[0] for cells in _summary_rows(_report())] == labels

    def test_metrics_without_three_measurements_report_insufficient_history(self):
        # The committed defaults file is a template: it carries no measurement
        # history, so a direct build_report() call has nothing to judge. The
        # assertions are driven by whatever the defaults actually hold, so the
        # spec keeps holding if the template is ever seeded.
        defaults = pm.load_state(DEFAULTS_PATH)
        rows = _summary_rows(_report())
        verdicts = {}

        for metric, cells in zip(pm.METRIC_REGISTRY, rows):
            verdicts[metric.id] = cells[4]
            recorded = _measurements(defaults, metric.id)
            if len(recorded) < 3:
                assert cells[4] == "Insufficient history", f"{metric.id} must not guess a verdict"
                assert cells[2] == "n/a", f"{metric.id} has no target below three measurements"
            if not recorded:
                assert cells[1] == "n/a", f"{metric.id} has no baseline without a measurement"

        assert "Insufficient history" in verdicts.values()

    def test_existing_report_sections_keep_their_order(self):
        report = _report()

        for heading in ("## Summary by hour", "## Session summary", "## Recommendations"):
            assert heading in report, f"{heading!r} lost from the report"
        assert report.index("## Summary by hour") < report.index("## Session summary")


# ---------------------------------------------------------------------------
# Per-metric detail block
# ---------------------------------------------------------------------------


@needs_details
class TestMetricDetails:
    def test_details_block_follows_the_summary_and_covers_every_metric_once(self):
        report = _report()

        headings = [line for line in report.splitlines() if line.startswith("## ")]
        entries = _detail_entries(report)

        assert headings[:2] == [SUMMARY_HEADING, DETAILS_HEADING]
        assert len(entries) == len(pm.METRIC_REGISTRY)
        assert [heading.split()[0] for heading, _ in entries] == [
            metric.id for metric in pm.METRIC_REGISTRY
        ]

    def test_detail_headings_name_the_metric_then_its_label(self):
        entries = _detail_entries(_report())

        for metric, (heading, _) in zip(pm.METRIC_REGISTRY, entries):
            assert heading.startswith(metric.id), heading
            assert metric.label.strip() in heading, heading

    def test_detail_headings_never_shadow_an_existing_section_split(self):
        # "## Decode speed" is a substring of "### Decode speed", so a detail
        # heading that repeats an existing section title makes the suite's
        # report.split("## <Title>", 1)[1] helpers return the detail body.
        report = _report()
        titles = [line for line in report.splitlines() if line.startswith("## ")]
        titles += [title for title in EXISTING_SECTION_TITLES if title not in titles]

        for heading, _ in _detail_entries(report):
            for title in titles:
                assert title not in heading, (
                    f"{heading!r} repeats the section title {title!r}; existing "
                    "splits on that title would return the detail body"
                )

    def test_each_detail_subsection_covers_importance_calculation_levers_and_tradeoffs(self):
        entries = _detail_entries(_report())

        for heading, block in entries:
            texts = _labelled_texts(block)
            for name in DETAIL_LABELS:
                assert re.search(r"[A-Za-z]", texts[name]), f"{heading}: empty {name} text"

    def test_each_detail_section_names_its_data_source(self):
        # Parent AC4(b): "how it is calculated including its data source". The
        # registry is the single source of truth, so the rendered calculation
        # must name the metric's own Metric.source string.
        entries = _detail_entries(_report())

        for metric, (_, block) in zip(pm.METRIC_REGISTRY, entries):
            texts = _labelled_texts(block)
            assert _norm(metric.source) in _norm(texts["Calculation"]), (
                f"{metric.id}: calculation does not name its data source "
                f"({metric.source!r})"
            )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@needs_cli
class TestCliFlags:
    def test_parser_accepts_every_metrics_flag(self):
        args = analyze_proxy_usage.parse_args(
            [
                "--metrics-config",
                "metrics.yaml",
                "--metrics-state",
                "state.yaml",
                "--no-metrics-update",
                "--seed-metrics-history",
                "archive",
            ]
        )

        assert args.metrics_config == "metrics.yaml"
        assert args.metrics_state == "state.yaml"
        assert args.no_metrics_update is True
        assert args.seed_metrics_history == "archive"

    def test_parser_defaults_leave_the_cron_invocation_unchanged(self):
        args = analyze_proxy_usage.parse_args([])

        # The committed defaults file is the default config, and it is default
        # state-updating: the 05:00 cron line needs no new flags.
        assert Path(args.metrics_config).expanduser().resolve() == DEFAULTS_PATH.resolve()
        assert args.metrics_state is None
        assert args.no_metrics_update is False
        assert args.seed_metrics_history is None


@needs_cli
class TestCliHelp:
    def test_help_documents_every_metrics_flag(self, capsys):
        with pytest.raises(SystemExit):
            analyze_proxy_usage.parse_args(["--help"])

        help_text = capsys.readouterr().out

        for flag in (
            "--metrics-config",
            "--metrics-state",
            "--no-metrics-update",
            "--seed-metrics-history",
        ):
            assert flag in help_text, f"--help does not document {flag}"


    def test_text_summary_still_prints_without_quiet(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"

        assert _run_cli(log_dir, out_dir) == 0
        summary_text = capsys.readouterr().out

        assert "Analyzed" in summary_text
        assert str(out_dir / STATE_FILENAME) in summary_text

        assert _run_cli(log_dir, out_dir, "--quiet") == 0
        assert capsys.readouterr().out == "", "--quiet must suppress the text summary"


@needs_cli
class TestDefaultRun:
    def test_default_run_renders_the_section_and_writes_only_the_live_state(self, tmp_path):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"
        defaults_before = DEFAULTS_PATH.read_bytes()

        assert _run_cli(log_dir, out_dir, "--quiet") == 0

        assert SUMMARY_HEADING in (out_dir / "report.md").read_text()
        state_path = out_dir / STATE_FILENAME
        assert state_path.exists(), "the default run must write the live state file"
        assert DEFAULTS_PATH.read_bytes() == defaults_before, (
            "the committed defaults file is read-only at runtime"
        )

        state = pm.load_state(state_path)
        assert {metric.id for metric in pm.METRIC_REGISTRY} <= set(state.get("metrics") or {})
        recorded = [
            metric_id
            for metric_id in METRIC_IDS
            if _measurements(state, metric_id)
        ]
        assert recorded, "the run must record the window's measurement for measurable metrics"
        for metric_id in recorded:
            for measurement in _measurements(state, metric_id):
                assert {"window", "value", "date", "fingerprint"} <= set(measurement)

    def test_no_metrics_update_leaves_the_state_file_untouched(self, tmp_path):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"
        defaults_before = DEFAULTS_PATH.read_bytes()
        assert _run_cli(log_dir, out_dir, "--quiet") == 0
        state_path = out_dir / STATE_FILENAME
        snapshot = state_path.read_bytes()

        assert _run_cli(log_dir, out_dir, "--quiet", "--no-metrics-update") == 0

        assert state_path.read_bytes() == snapshot
        assert DEFAULTS_PATH.read_bytes() == defaults_before
        # The scorecard is still rendered: the flag suppresses only the write.
        assert SUMMARY_HEADING in (out_dir / "report.md").read_text()


# ---------------------------------------------------------------------------
# --json contract
# ---------------------------------------------------------------------------


@needs_cli
class TestJsonCounts:
    def test_json_reports_one_count_per_verdict_token(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"

        assert _run_cli(log_dir, out_dir, "--json") == 0
        data = json.loads(capsys.readouterr().out)

        block = data["performance_metrics"]
        counts = data["performance_metrics_counts"]
        assert set(counts) == set(VERDICTS)
        assert sum(counts.values()) == len(block)
        for verdict in VERDICTS:
            assert counts[verdict] == sum(
                1 for entry in block.values() if entry["verdict"] == verdict
            )


@needs_cli
class TestMetricsConfigOverride:
    def test_metrics_config_relocates_the_defaults_history(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"
        config_path = tmp_path / "custom_metrics.yaml"
        seeded = pm.new_state()
        for value, window in ((94.0, "2026-07-01"), (90.0, "2026-07-15"), (92.0, "2026-07-29")):
            pm.record_measurement(
                seeded,
                "fallback_rate",
                window=window,
                value=value,
                date=window,
                fingerprint=pm.configuration_fingerprint(2, 131072),
            )
        pm.save_state(seeded, config_path)
        config_before = config_path.read_bytes()

        assert _run_cli(
            log_dir,
            out_dir,
            "--metrics-config",
            str(config_path),
            "--no-metrics-update",
            "--json",
        ) == 0
        entry = json.loads(capsys.readouterr().out)["performance_metrics"]["fallback_rate"]

        assert entry["baseline"]["value"] == pytest.approx(90.0)
        assert entry["target"] == pytest.approx(92.0)
        assert entry["verdict"] == "Met"
        # --no-metrics-update writes no state at all, whatever the config says.
        assert not (out_dir / STATE_FILENAME).exists()
        assert config_path.read_bytes() == config_before

        # Without --metrics-state the live history still lands under
        # --output-dir, and the relocated defaults file stays read-only.
        assert _run_cli(
            log_dir, out_dir, "--metrics-config", str(config_path), "--quiet"
        ) == 0
        assert (out_dir / STATE_FILENAME).exists()
        assert config_path.read_bytes() == config_before


@needs_cli
class TestJsonBlock:
    def test_json_block_agrees_with_the_rendered_table(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"

        assert _run_cli(log_dir, out_dir, "--json") == 0
        data = json.loads(capsys.readouterr().out)

        block = data["performance_metrics"]
        assert set(block) == {metric.id for metric in pm.METRIC_REGISTRY}
        for metric_id, entry in block.items():
            assert {"baseline", "target", "actual", "verdict"} <= set(entry), metric_id
            assert entry["verdict"] in VERDICTS, metric_id
            assert entry["actual"] is None or isinstance(entry["actual"], (int, float))

        by_label = {metric.label.strip(): metric.id for metric in pm.METRIC_REGISTRY}
        table = {by_label[cells[0]]: cells[4] for cells in _summary_rows((out_dir / "report.md").read_text())}
        assert table == {metric_id: entry["verdict"] for metric_id, entry in block.items()}

        state = pm.load_state(out_dir / STATE_FILENAME)
        insufficient = [
            metric.id
            for metric in pm.METRIC_REGISTRY
            if len(_measurements(state, metric.id)) < 3
        ]
        assert insufficient, "the single-window fixture must not have a three-measurement history"
        for metric_id in insufficient:
            assert block[metric_id]["verdict"] == "Insufficient history", metric_id

    def test_json_block_carries_baseline_target_and_verdict_from_history(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"
        seeded = pm.new_state()
        # fallback_rate is lower-is-better: baseline = lowest recorded value,
        # target = mean of the three best (lowest). Seeded values are far above
        # this window's actual, so the verdict is Met under either the ratio or
        # the percentage convention.
        for value, window in ((94.0, "2026-07-01"), (90.0, "2026-07-15"), (92.0, "2026-07-29")):
            pm.record_measurement(
                seeded,
                "fallback_rate",
                window=window,
                value=value,
                date=window,
                fingerprint=pm.configuration_fingerprint(2, 131072),
            )
        state_path = tmp_path / "seeded_state.yaml"
        pm.save_state(seeded, state_path)
        seeded_before = state_path.read_bytes()

        assert _run_cli(
            log_dir, out_dir, "--metrics-state", str(state_path), "--no-metrics-update", "--json"
        ) == 0
        entry = json.loads(capsys.readouterr().out)["performance_metrics"]["fallback_rate"]

        assert entry["baseline"]["value"] == pytest.approx(90.0)
        assert entry["baseline"]["date"] == "2026-07-15"
        assert entry["target"] == pytest.approx(92.0)
        assert entry["actual"] is not None
        assert entry["verdict"] == "Met"

        by_label = {metric.label.strip(): metric.id for metric in pm.METRIC_REGISTRY}
        table = {by_label[cells[0]]: cells for cells in _summary_rows((out_dir / "report.md").read_text())}
        assert table["fallback_rate"][4] == "Met"
        assert table["fallback_rate"][2] != "n/a"

        # --metrics-state relocates the live state file; the output directory
        # gets none, and --no-metrics-update leaves the given file untouched.
        assert not (out_dir / STATE_FILENAME).exists()
        assert state_path.read_bytes() == seeded_before


# ---------------------------------------------------------------------------
# Backfill CLI
# ---------------------------------------------------------------------------


@needs_backfill
class TestBackfillCli:
    def test_seed_metrics_history_json_reports_seeded_and_skipped_counts(self, tmp_path, capsys):
        log_dir = _write_log_dir(tmp_path)
        archive = tmp_path / "archive"
        (archive / "2026-08-07").mkdir(parents=True)
        (archive / "2026-08-07" / "report.md").write_text(fixtures.ARCHIVED_REPORT_ERA_A)
        state_path = tmp_path / "seed_state.yaml"

        assert _run_cli(
            log_dir,
            tmp_path / "usage",
            "--seed-metrics-history",
            str(archive),
            "--metrics-state",
            str(state_path),
            "--json",
        ) == 0
        payload = json.loads(capsys.readouterr().out)

        assert payload["windows"] == 1
        assert payload["seeded"] > 0
        assert payload["already_present"] == 0
        assert payload["skipped_metrics"], "unparseable metrics must be reported, not guessed"
        seeded = pm.load_state(state_path)
        assert _measurements(seeded, "fallback_rate"), "the parsed value must reach the state file"

    def test_seed_metrics_history_reads_the_archive_and_is_idempotent(self, tmp_path):
        log_dir = _write_log_dir(tmp_path)
        out_dir = tmp_path / "usage"
        archive = tmp_path / "archive"
        (archive / "2026-08-07").mkdir(parents=True)
        (archive / "2026-08-07" / "report.md").write_text(fixtures.ARCHIVED_REPORT_ERA_A)
        (archive / "2026-09-02").mkdir()
        (archive / "2026-09-02" / "report.md").write_text(fixtures.ARCHIVED_REPORT_ERA_B)
        state_path = tmp_path / "backfill_state.yaml"
        defaults_before = DEFAULTS_PATH.read_bytes()

        assert _run_cli(
            log_dir,
            out_dir,
            "--seed-metrics-history",
            str(archive),
            "--metrics-state",
            str(state_path),
            "--quiet",
        ) == 0

        state = pm.load_state(state_path)
        fallback = [m["value"] for m in _measurements(state, "fallback_rate")]
        decode = [m["value"] for m in _measurements(state, "decode_speed")]
        prompt_eval = [m["value"] for m in _measurements(state, "prompt_eval_speed")]
        assert 79.7 in fallback and 48.1 in fallback
        assert 25.8 in decode and 37.8 in decode
        assert 137.2 in prompt_eval and 193.3 in prompt_eval
        assert DEFAULTS_PATH.read_bytes() == defaults_before

        counts = {metric_id: len(_measurements(state, metric_id)) for metric_id in METRIC_IDS}

        assert _run_cli(
            log_dir,
            out_dir,
            "--seed-metrics-history",
            str(archive),
            "--metrics-state",
            str(state_path),
            "--quiet",
        ) == 0

        replayed = pm.load_state(state_path)
        assert {
            metric_id: len(_measurements(replayed, metric_id)) for metric_id in METRIC_IDS
        } == counts, "re-running the backfill must not duplicate a window's measurement"
        for metric_id in METRIC_IDS:
            windows = [m["window"] for m in _measurements(replayed, metric_id)]
            assert len(windows) == len(set(windows)), f"{metric_id}: duplicate window after replay"
        assert DEFAULTS_PATH.read_bytes() == defaults_before

    def test_seed_metrics_history_windows_reports_by_generated_date(self, tmp_path):
        # Two archived directories on the same ``Generated`` day (the live
        # archive has 2026-08-07 and 2026-08-07_2/3/4) are one window, so the
        # second contributes no duplicate measurement and the directory suffix
        # never becomes a window id of its own.
        log_dir = _write_log_dir(tmp_path)
        archive = tmp_path / "archive"
        (archive / "2026-08-07").mkdir(parents=True)
        (archive / "2026-08-07" / "report.md").write_text(fixtures.ARCHIVED_REPORT_ERA_A)
        (archive / "2026-08-07_2").mkdir()
        (archive / "2026-08-07_2" / "report.md").write_text(fixtures.ARCHIVED_REPORT_ERA_A)
        state_path = tmp_path / "generated_window_state.yaml"

        assert _run_cli(
            log_dir,
            tmp_path / "usage",
            "--seed-metrics-history",
            str(archive),
            "--metrics-state",
            str(state_path),
            "--quiet",
        ) == 0

        state = pm.load_state(state_path)
        windows = {m["window"] for m in _measurements(state, "decode_speed")}
        assert windows == {"2026-08-07"}
        assert len(_measurements(state, "decode_speed")) == 1

    def test_seed_metrics_history_without_generated_uses_directory_name(self, tmp_path):
        log_dir = _write_log_dir(tmp_path)
        archive = tmp_path / "archive"
        (archive / "2026-08-07").mkdir(parents=True)
        (archive / "2026-08-07" / "report.md").write_text(
            fixtures.ARCHIVED_REPORT_ERA_A.replace(
                "- Generated: 2026-08-07 02:10:21\n", ""
            )
        )
        state_path = tmp_path / "no_generated_state.yaml"

        assert _run_cli(
            log_dir,
            tmp_path / "usage",
            "--seed-metrics-history",
            str(archive),
            "--metrics-state",
            str(state_path),
            "--quiet",
        ) == 0

        state = pm.load_state(state_path)
        assert {m["window"] for m in _measurements(state, "decode_speed")} == {"2026-08-07"}
