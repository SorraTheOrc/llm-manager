"""Nightly ``slots_stale`` capacity signals (LP-0MVU8O4EL005IYP9).

The 05:00 nightly report must publish the stale-poll rate for its own window
against a configurable target, derived from the **same** implementation as the
standalone ``scripts/capacity_signal_verification.py`` tool, so the operator no
longer has to run that tool by hand and cross-check two surfaces.

This module verifies the whole vertical slice:

- AC1.1 the derivation lives once, in ``scripts/lib/capacity_signals.py``: the
  reference tool's status-poll detection / target check come from that module
  and the nightly analyzer reaches it through the documented path shim;
- AC1.2 the parser carries ``slots_stale``, ``slots_detail_stale`` and
  ``latency_ms`` on ``status_request`` events only, ``None`` on pre-change
  (schema-1) lines;
- AC1.3 status polls are aggregated with ``[window_start, window_end)``
  semantics and report polls / stale / denominator / rate;
- AC1.4 the report renders the ``## Local capacity signals`` section with an
  explicit met/unmet outcome, and omits it entirely on a window with no polls;
- AC1.5 ``--json`` gains ``capacity_signals.slots_stale`` while every existing
  key keeps its meaning;
- AC1.6 ``--stale-rate-target`` changes both the rendered target and the
  pass/fail outcome;
- AC1.7 (this module) a divergence test that fails if the nightly analyzer and
  the reference tool disagree on the same synthetic lines;
- AC1.8 an empty window — or a log directory with no logs at all — renders a
  clean report and exits 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

REPO_ROOT = Path(__file__).resolve().parents[4]

import aggregation  # noqa: E402
import analyze_proxy_usage  # noqa: E402
import bucketing  # noqa: E402
import capacity_signals  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

from tests import fixtures  # noqa: E402

WINDOW_START = datetime(2026, 10, 28, 10, 0, 0)
WINDOW_END = datetime(2026, 10, 28, 11, 0, 0)

SECTION_HEADING = "## Local capacity signals"
CONTENTION_HEADING = "## Local contention prediction vs reality"


# ---------------------------------------------------------------------------
# Synthetic status-poll lines (one builder per schema shape)
# ---------------------------------------------------------------------------


def _poll_line(
    ts: str,
    *,
    schema: int,
    hint: str = "idle",
    available_slots: int = 1,
    stale: bool | None = None,
    detail: bool | None = None,
    latency_ms: int | None = None,
    in_flight: int | None = None,
    local_active: bool = False,
) -> str:
    """A ``status_request`` line in the requested schema shape.

    A field left ``None`` is *absent* from the line — which is how a pre-change
    line looks, and the point of the tri-state parsing.
    """
    fields = [
        f"active_query={'true' if local_active else 'false'}",
        f"available_slots={available_slots}",
        f"capacity_hint={hint}",
        f"local_active_query={'true' if local_active else 'false'}",
    ]
    if in_flight is not None:
        fields.append(f"local_in_flight_requests={in_flight}")
    if latency_ms is not None:
        fields.append(f"latency_ms={latency_ms}")
    if detail is not None:
        fields.append(f"slots_detail_stale={'true' if detail else 'false'}")
    if stale is not None:
        fields.append(f"slots_stale={'true' if stale else 'false'}")
    fields.append(f"status_schema_version={schema}")
    fields.append("total_slots=2")
    return f"{ts} - INFO - status_request " + " ".join(fields)


def _schema1(ts: str, *, hint: str = "idle") -> str:
    """Pre-change poll: no ``slots_stale`` / ``slots_detail_stale`` / latency."""
    return _poll_line(ts, schema=1, hint=hint)


def _schema2(ts: str, *, stale: bool, hint: str = "idle") -> str:
    """Schema 2: carries ``slots_stale`` but no per-source flag or in-flight count."""
    return _poll_line(ts, schema=2, stale=stale, hint=hint)


def _schema3(
    ts: str,
    *,
    stale: bool,
    detail: bool | None = None,
    hint: str = "idle",
    latency_ms: int = 4,
    in_flight: int = 0,
) -> str:
    """Schema 3: the full payload, including per-source freshness and latency."""
    return _poll_line(
        ts,
        schema=3,
        stale=stale,
        detail=stale if detail is None else detail,
        hint=hint,
        latency_ms=latency_ms,
        in_flight=in_flight,
    )


# ---------------------------------------------------------------------------
# Helpers: aggregation, report and the reference tool
# ---------------------------------------------------------------------------


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "proxy.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_dir


def _aggregate(
    log_dir: Path,
    window_start: datetime = WINDOW_START,
    window_end: datetime = WINDOW_END,
) -> aggregation.AnalysisResult:
    """Run the nightly pipeline's parse + aggregate over *log_dir*."""
    events = list(
        log_parser.iter_events(log_dir / "proxy.log", window_start, window_end)
    )
    schedule = bucketing.schedule_from_entries([("23:59", 8)])
    return aggregation.aggregate(
        events, window_start, window_end, schedule, None
    )


def _analyse(lines: list[str], tmp_path: Path) -> aggregation.AnalysisResult:
    return _aggregate(_write_log(tmp_path, lines))


def _report(
    summary: aggregation.AnalysisResult,
    stale_rate_target: float = capacity_signals.DEFAULT_STALE_RATE_TARGET,
) -> str:
    return reporting.build_report(
        summary, None, stale_rate_target=stale_rate_target
    )


def _load_reference_tool():
    """Import ``scripts/capacity_signal_verification.py`` by file path.

    The tool is a repo-level script, not part of the skill's package, so it is
    loaded the same way ``proxy/tests/test_capacity_signal_verification.py``
    loads it.
    """
    path = REPO_ROOT / "scripts" / "capacity_signal_verification.py"
    spec = importlib.util.spec_from_file_location(
        "capacity_signal_verification_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_tool():
    return _load_reference_tool()


@pytest.fixture(autouse=True)
def _no_prometheus(monkeypatch):
    """Keep the end-to-end CLI tests hermetic (no Prometheus round-trip)."""
    monkeypatch.setattr(
        reporting.gpu_busy, "compute_gpu_busy", lambda *args, **kwargs: None
    )


# ---------------------------------------------------------------------------
# AC1.1 — one derivation, imported by both surfaces
# ---------------------------------------------------------------------------


class TestSingleDerivation:
    def test_nightly_shim_loads_the_repo_level_module(self):
        expected = REPO_ROOT / "scripts" / "lib" / "capacity_signals.py"
        assert capacity_signals.SHARED_PATH == expected
        assert expected.is_file()
        # Provenance, not a copy: the nightly surface's rule is the module that
        # was loaded from the shared file above.
        assert capacity_signals.stale_rate_check.__module__ == (
            "_repo_capacity_signals"
        )
        assert capacity_signals.is_status_poll.__module__ == (
            "_repo_capacity_signals"
        )

    def test_reference_tool_imports_the_shared_derivation(self, reference_tool):
        assert reference_tool.is_status_poll.__module__ == "lib.capacity_signals"
        assert reference_tool.has_live_local_work.__module__ == (
            "lib.capacity_signals"
        )
        assert reference_tool.stale_rate_check.__module__ == (
            "lib.capacity_signals"
        )
        # The private copies are gone, so there is nothing to drift from.
        assert not hasattr(reference_tool, "_status_fields")
        assert not hasattr(reference_tool, "_STATUS_FIELD_RE")

    def test_tally_excludes_polls_without_the_field(self):
        # AC1.1: a poll with no ``slots_stale`` field never counts as fresh.
        assert capacity_signals.tally_stale([True, False, None, True]) == (2, 3)
        assert capacity_signals.tally_stale([]) == (0, 0)

    def test_rate_and_target_check(self):
        assert capacity_signals.stale_rate(0, 0) == 0.0
        assert capacity_signals.stale_rate(1, 3) == 0.3333
        # Strict comparison: a rate exactly at the target is a miss.
        assert capacity_signals.stale_rate_check(5, 100, 0.05)["passed"] is False
        assert capacity_signals.stale_rate_check(4, 100, 0.05)["passed"] is True
        assert capacity_signals.stale_rate_check(4, 100, 0.05) == {
            "target": 0.05,
            "observed": 0.04,
            "denominator": 100,
            "passed": True,
        }

    def test_half_open_window_rule(self):
        assert capacity_signals.in_window(WINDOW_START, WINDOW_START, WINDOW_END)
        assert not capacity_signals.in_window(WINDOW_END, WINDOW_START, WINDOW_END)
        assert not capacity_signals.in_window(
            WINDOW_START - timedelta(seconds=1), WINDOW_START, WINDOW_END
        )
        assert not capacity_signals.in_window(None, WINDOW_START, WINDOW_END)


# ---------------------------------------------------------------------------
# AC1.2 — parser fields
# ---------------------------------------------------------------------------


class TestParser:
    def test_schema1_poll_has_no_freshness_fields(self):
        event = log_parser.parse_log_line(_schema1("2026-10-28 10:05:00,000"))
        assert event.kind == "status_request"
        assert event.capacity_hint == "idle"
        assert event.status_schema_version == 1
        # Absent, never fabricated as "fresh".
        assert event.slots_stale is None
        assert event.slots_detail_stale is None
        assert event.latency_ms is None

    def test_schema2_poll_parses_the_union_stale_flag(self):
        stale = log_parser.parse_log_line(
            _schema2("2026-10-28 10:05:00,000", stale=True, hint="saturated")
        )
        fresh = log_parser.parse_log_line(
            _schema2("2026-10-28 10:06:00,000", stale=False)
        )
        assert (stale.slots_stale, stale.capacity_hint) == (True, "saturated")
        assert fresh.slots_stale is False
        # The per-source flag is a schema-3 field: still absent here.
        assert stale.slots_detail_stale is None
        assert stale.latency_ms is None

    def test_schema3_poll_parses_detail_and_latency(self):
        event = log_parser.parse_log_line(
            _schema3(
                "2026-10-28 10:05:00,000",
                stale=True,
                detail=False,
                latency_ms=3024,
                in_flight=2,
            )
        )
        assert event.kind == "status_request"
        assert event.slots_stale is True
        assert event.slots_detail_stale is False
        assert event.latency_ms == 3024
        assert event.status_schema_version == 3
        # Existing status fields are untouched.
        assert event.available_slots == 1
        assert event.total_slots == 2

        fresh = log_parser.parse_log_line(
            _schema3("2026-10-28 10:06:00,000", stale=False, latency_ms=2)
        )
        assert fresh.slots_stale is False
        assert fresh.slots_detail_stale is False
        assert fresh.latency_ms == 2

    def test_other_event_kinds_gain_no_freshness_fields(self):
        stream = log_parser.parse_log_line(fixtures.STREAM_STARTED_LOCAL)
        assert stream is not None and stream.kind == "stream_started"
        assert stream.slots_stale is None
        assert stream.slots_detail_stale is None
        assert stream.latency_ms is None

    def test_a_line_without_capacity_hint_is_not_a_status_poll(self):
        # ``is_status_poll`` requires the ``capacity_hint=`` field, so a
        # malformed/partial status line cannot inflate the poll count.
        line = "2026-10-28 10:05:00,000 - INFO - status_request available_slots=1"
        assert capacity_signals.is_status_poll(line) is False


# ---------------------------------------------------------------------------
# AC1.3 — half-open aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_poll_at_window_start_counts_and_poll_at_end_is_excluded(self, tmp_path):
        summary = _analyse(
            [
                _schema3("2026-10-28 10:00:00,000", stale=True),
                _schema3("2026-10-28 10:30:00,000", stale=False),
                _schema3("2026-10-28 11:00:00,000", stale=True),
            ],
            tmp_path,
        )
        signals = summary.capacity_signals
        assert signals.status_polls == 2
        assert signals.slots_stale == 1
        assert signals.slots_stale_denominator == 2
        assert signals.slots_stale_rate == 0.5

    def test_schema1_polls_are_counted_but_excluded_from_the_denominator(
        self, tmp_path
    ):
        summary = _analyse(
            [
                _schema1("2026-10-28 10:00:00,000"),
                _schema2("2026-10-28 10:10:00,000", stale=True),
                _schema1("2026-10-28 10:20:00,000"),
            ],
            tmp_path,
        )
        signals = summary.capacity_signals
        assert signals.status_polls == 3
        assert signals.slots_stale_denominator == 1
        assert signals.slots_stale == 1
        assert signals.slots_stale_rate == 1.0

    def test_non_status_events_do_not_enter_the_aggregation(self, tmp_path):
        summary = _analyse(
            [
                fixtures.STREAM_STARTED_LOCAL.replace(
                    "2026-08-02 13:58:32", "2026-10-28 10:15:00"
                ),
                _schema3("2026-10-28 10:30:00,000", stale=False),
            ],
            tmp_path,
        )
        assert summary.capacity_signals.status_polls == 1
        assert summary.capacity_signals.slots_stale_rate == 0.0


# ---------------------------------------------------------------------------
# AC1.4 / AC1.6 — the rendered section and its configurable target
# ---------------------------------------------------------------------------


class TestReportSection:
    def _summary(self, tmp_path: Path) -> aggregation.AnalysisResult:
        return _analyse(
            [
                _schema3("2026-10-28 10:00:00,000", stale=True),
                _schema3("2026-10-28 10:30:00,000", stale=False),
            ],
            tmp_path,
        )

    def test_section_renders_counts_rate_target_and_outcome(self, tmp_path):
        report = _report(self._summary(tmp_path))
        assert SECTION_HEADING in report
        # Immediately after the contention-prediction section it explains.
        assert report.index(SECTION_HEADING) > report.index(CONTENTION_HEADING)
        assert "| Status polls (`status_request` with `capacity_hint=`) | 2 |" in report
        assert "| Polls with `slots_stale=true` | 1 |" in report
        assert "| Stale rate | 50.00% |" in report
        assert "| Target | < 5.00% of status polls |" in report
        assert "| Outcome | **unmet** |" in report

    def test_target_override_changes_the_rendered_target_and_outcome(self, tmp_path):
        summary = self._summary(tmp_path)
        permissive = _report(summary, stale_rate_target=0.9)
        assert "| Target | < 90.00% of status polls |" in permissive
        assert "| Outcome | **met** |" in permissive
        assert "| Outcome | **unmet** |" in _report(summary, stale_rate_target=0.05)

    def test_section_is_omitted_when_the_window_has_no_status_polls(self, tmp_path):
        summary = _analyse(
            [
                fixtures.STREAM_STARTED_LOCAL.replace(
                    "2026-08-02 13:58:32", "2026-10-28 10:15:00"
                )
            ],
            tmp_path,
        )
        report = _report(summary)
        assert SECTION_HEADING not in report
        assert "slots_stale" not in report

    def test_section_states_the_excluded_schema1_polls(self, tmp_path):
        summary = _analyse(
            [
                _schema1("2026-10-28 10:00:00,000"),
                _schema3("2026-10-28 10:30:00,000", stale=True),
            ],
            tmp_path,
        )
        report = _report(summary)
        assert "1 poll(s) carry no `slots_stale` field (schema 1)" in report


# ---------------------------------------------------------------------------
# AC1.5 — the additive JSON key
# ---------------------------------------------------------------------------


class TestJson:
    def test_capacity_signals_block_mirrors_the_section(self, tmp_path):
        summary = _analyse(
            [
                _schema3("2026-10-28 10:00:00,000", stale=True),
                _schema3("2026-10-28 10:30:00,000", stale=False),
            ],
            tmp_path,
        )
        payload = reporting.summary_to_json(summary, None)
        assert payload["capacity_signals"]["slots_stale"] == {
            "status_polls": 2,
            "count": 1,
            "denominator": 2,
            "rate": 0.5,
            "target": 0.05,
            "passed": False,
        }

    def test_existing_keys_keep_their_meaning(self, tmp_path):
        summary = _analyse(
            [_schema3("2026-10-28 10:00:00,000", stale=False)], tmp_path
        )
        payload = reporting.summary_to_json(summary, None)
        # A representative slice of the pre-existing surface: still present and
        # still computed from the summary (nothing renamed or re-scoped).
        assert payload["total_requests"] == summary.total_requests
        assert payload["window_start"] == "2026-10-28 10:00:00"
        assert payload["window_end"] == "2026-10-28 11:00:00"
        assert "status_capacity_hint" in payload
        assert "local_turn_coverage" in payload

    def test_target_override_is_reflected_in_the_json_verdict(self, tmp_path):
        summary = _analyse(
            [
                _schema3("2026-10-28 10:00:00,000", stale=True),
                _schema3("2026-10-28 10:30:00,000", stale=False),
            ],
            tmp_path,
        )
        payload = reporting.summary_to_json(summary, None, stale_rate_target=0.9)
        assert payload["capacity_signals"]["slots_stale"]["target"] == 0.9
        assert payload["capacity_signals"]["slots_stale"]["passed"] is True

    def test_empty_window_reports_no_outcome_instead_of_a_false_all_clear(
        self, tmp_path
    ):
        summary = _analyse([], tmp_path)
        block = reporting.summary_to_json(summary, None)["capacity_signals"][
            "slots_stale"
        ]
        assert block["status_polls"] == 0
        assert block["passed"] is None


# ---------------------------------------------------------------------------
# AC1.7 — the two surfaces agree on the same synthetic lines
# ---------------------------------------------------------------------------


class TestDivergence:
    LINES = [
        _schema1("2026-10-28 10:00:00,000"),
        _schema3("2026-10-28 10:00:00,500", stale=True),
        _schema2("2026-10-28 10:10:00,000", stale=False),
        _schema3("2026-10-28 10:20:00,000", stale=True, hint="saturated"),
        _schema2("2026-10-28 10:30:00,000", stale=True),
        _schema3("2026-10-28 10:40:00,000", stale=False),
        # Exactly at ``window_end``: both surfaces must exclude it.
        _schema3("2026-10-28 11:00:00,000", stale=True),
    ]

    def test_nightly_and_reference_tool_publish_identical_figures(
        self, tmp_path, reference_tool
    ):
        log_dir = _write_log(tmp_path, self.LINES)
        nightly = _aggregate(log_dir).capacity_signals
        reference = reference_tool.analyse_status_stream(
            log_dir, since=WINDOW_START, until=WINDOW_END
        )

        assert reference["status_polls"] == nightly.status_polls
        assert reference["slots_stale"]["count"] == nightly.slots_stale
        assert (
            reference["slots_stale"]["denominator"]
            == nightly.slots_stale_denominator
        )
        assert reference["slots_stale"]["rate"] == nightly.slots_stale_rate
        # Sanity: the fixture is not degenerate. Six polls fall in the window,
        # the schema-1 one has no freshness field (so five are eligible) and
        # three of those are stale.
        assert nightly.status_polls == 6
        assert nightly.slots_stale == 3
        assert nightly.slots_stale_denominator == 5
        assert nightly.slots_stale_rate == 0.6

    def test_agreement_holds_for_an_override_target(
        self, tmp_path, reference_tool
    ):
        log_dir = _write_log(tmp_path, self.LINES)
        nightly = reporting.capacity_signals_json(
            _aggregate(log_dir), stale_rate_target=0.4
        )["slots_stale"]
        reference = reference_tool.evaluate(
            reference_tool.analyse_status_stream(
                log_dir, since=WINDOW_START, until=WINDOW_END
            ),
            {"gaps": [], "covered": True},
            {"covered": False, "count": 0, "note": "not retained"},
            stale_rate_target=0.4,
        )
        check = next(
            c for c in reference["checks"] if "slots_stale" in c["target"]
        )
        assert check["observed"] == nightly["rate"]
        assert check["pass"] is nightly["passed"]


# ---------------------------------------------------------------------------
# AC1.6 / AC1.8 — the CLI end to end
# ---------------------------------------------------------------------------


class TestCli:
    def _run(self, log_dir: Path, out_dir: Path, *extra: str) -> int:
        return analyze_proxy_usage.main(
            [
                "--log-dir",
                str(log_dir),
                "--start",
                "2026-10-28 10:00:00",
                "--end",
                "2026-10-28 11:00:00",
                "--output-dir",
                str(out_dir),
                *extra,
            ]
        )

    def test_stale_rate_target_flag_reaches_report_and_json(
        self, tmp_path, capsys
    ):
        log_dir = _write_log(
            tmp_path,
            [
                _schema3("2026-10-28 10:00:00,000", stale=True),
                _schema3("2026-10-28 10:30:00,000", stale=False),
            ],
        )
        out_dir = tmp_path / "out"
        assert self._run(log_dir, out_dir, "--json", "--stale-rate-target", "0.9") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["capacity_signals"]["slots_stale"]["target"] == 0.9
        assert payload["capacity_signals"]["slots_stale"]["passed"] is True
        report = (out_dir / "report.md").read_text(encoding="utf-8")
        assert "| Target | < 90.00% of status polls |" in report
        assert "| Outcome | **met** |" in report

    def test_empty_log_directory_renders_a_clean_report_and_exits_zero(
        self, tmp_path, capsys
    ):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        out_dir = tmp_path / "out"
        assert self._run(log_dir, out_dir, "--json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["capacity_signals"]["slots_stale"]["status_polls"] == 0
        assert payload["capacity_signals"]["slots_stale"]["passed"] is None
        report = (out_dir / "report.md").read_text(encoding="utf-8")
        assert SECTION_HEADING not in report

    def test_window_with_logs_but_no_status_polls_omits_the_section(
        self, tmp_path, capsys
    ):
        log_dir = _write_log(
            tmp_path,
            [
                fixtures.STREAM_STARTED_LOCAL.replace(
                    "2026-08-02 13:58:32", "2026-10-28 10:15:00"
                )
            ],
        )
        out_dir = tmp_path / "out"
        assert self._run(log_dir, out_dir, "--json") == 0
        capsys.readouterr()
        report = (out_dir / "report.md").read_text(encoding="utf-8")
        assert SECTION_HEADING not in report
