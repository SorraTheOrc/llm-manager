#!/usr/bin/env python3
"""Unit tests for ``scripts/capacity_signal_verification.py`` (LP-0MVOG6YRA000T9PK).

The tool turns a proxy-log window into the capacity figures the acceptance
criteria of LP-0MVLNBCTN0084MT0 are written against. These tests import the
script module directly and drive it with synthetic logs, so they assert the
observable measurement contract without needing the production log directory:

* status-poll parsing and the ``slots_stale`` rate (schema-aware denominator);
* the ``capacity_hint`` histogram;
* saturated-without-live-work detection, including the schema-2 fallback that
  the pre-fix false-``saturated`` incident was about, and the reported worst
  timestamps;
* the ``capacity_hint=saturated`` + ``available_slots=0`` subset that the
  recorded pre-fix control figure (82.3 %) was derived from;
* local dispatch counting (cache state, distinct sessions, herdr subset);
* herdr downtime dispatch counting from the JSONL log: UTC→local conversion,
  de-duplication of herdr's enriched second row, and an honest
  "not retained" answer when the rolling log does not cover the window;
* window → log-file mapping through ``scripts/lib/proxy_logs.py``: dot, dash
  and ``.gz`` names, per-line window boundaries, and the coverage-gap verdict
  (inside the window, at its head, and — only when the window is closed — at
  its tail) that must never let a partial read look like a measurement;
* the CLI contract: exit 0 for a covered (even empty) window, exit 3 for a
  coverage gap, and the ``--json`` document shape.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "capacity_signal_verification.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "capacity_signal_verification", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def csv():
    """The tool module under test (imported once per module)."""
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic log builders
# ---------------------------------------------------------------------------


def status_line(
    ts: str,
    *,
    hint: str = "idle",
    stale: str = "false",
    available: str = "2",
    local_active: str = "false",
    active: str = "false",
    in_flight: int | None = None,
    schema: int = 3,
) -> str:
    """One ``status_request`` log line, as the proxy writes it."""
    fields = [
        f"active_query={active}",
        f"available_slots={available}",
        f"capacity_hint={hint}",
        "current_model=Qwen3",
        "latency_ms=3",
        "llama_server_running=true",
        f"local_active_query={local_active}",
    ]
    if in_flight is not None:
        fields.append(f"local_in_flight_requests={in_flight}")
    fields += [
        "model_switch_in_progress=false",
        f"slots_stale={stale}",
        f"status_schema_version={schema}",
        "total_slots=2",
    ]
    return f"{ts},000 - INFO - status_request " + " ".join(fields) + "\n"


def dispatch_line(
    ts: str, *, session: str = "s1", cache_state: str = "warm"
) -> str:
    """One local dispatch completion line (``dispatch_first_byte_ms=``)."""
    return (
        f"{ts},000 - INFO - dispatch_first_byte_ms=1200.0 "
        f"dispatch_to_first_byte_ms=1199.0 cache_state={cache_state} "
        f"cached_ratio=0.9 queue_wait_ms=1.0 session={session} model=Qwen3\n"
    )


def write_log(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def utc_z(local_ts: dt.datetime) -> str:
    """Render a naive *local* wall clock as the UTC ``Z`` herdr writes."""
    return local_ts.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def downtime_log(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Status polls, slots_stale, capacity_hint
# ---------------------------------------------------------------------------


def test_status_poll_slots_stale_rate_and_hint_histogram(csv, tmp_path):
    """Polls, the stale rate (schema-aware) and the hint distribution."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line("2026-10-23 00:00:01", hint="saturated", stale="true")
        + status_line("2026-10-23 00:00:02", hint="saturated", stale="true")
        + status_line("2026-10-23 00:00:03", hint="idle", stale="false")
        + status_line("2026-10-23 00:00:04", hint="idle", stale="false")
        # Schema 1: no slots_stale field at all -> excluded from the rate,
        # never counted as "fresh".
        + "2026-10-23 00:00:05,000 - INFO - status_request capacity_hint=idle "
        "status_schema_version=1 total_slots=2\n",
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 24, 0, 0),
    )

    assert metrics["status_polls"] == 5
    assert metrics["slots_stale"] == {
        "count": 2,
        "denominator": 4,
        "rate": 0.5,
    }
    assert metrics["capacity_hint"]["histogram"] == {"idle": 3, "saturated": 2}
    assert metrics["capacity_hint"]["rates"] == {
        "idle": 0.6,
        "saturated": 0.4,
    }


def test_saturated_without_live_work_uses_inflight_then_flag_fallback(
    csv, tmp_path
):
    """A violating poll must show no live work signal at all.

    Schema 3 carries ``local_in_flight_requests`` (authoritative); schema 2 has
    only the ``local_active_query``/``active_query`` flags, which is exactly
    the surface the pre-fix false-``saturated`` incident was measured on.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        # Schema 3, live work in flight -> NOT a violation.
        status_line(
            "2026-10-23 00:00:01",
            hint="saturated",
            stale="true",
            in_flight=2,
            local_active="true",
        )
        # Schema 3, no live work -> violation.
        + status_line(
            "2026-10-23 00:00:02",
            hint="saturated",
            stale="true",
            in_flight=0,
        )
        # Schema 3, no live work, second violation in the same minute.
        + status_line(
            "2026-10-23 00:00:40",
            hint="saturated",
            stale="true",
            in_flight=0,
        )
        # Schema 2 (no in_flight field) with local_active_query=true -> not a
        # violation, the flag fallback must see it.
        + status_line(
            "2026-10-23 01:00:00",
            hint="saturated",
            stale="true",
            local_active="true",
            schema=2,
        )
        # Schema 2, flags false -> violation via the fallback.
        + status_line(
            "2026-10-23 02:10:00",
            hint="saturated",
            stale="true",
            schema=2,
        )
        # Not saturated -> never a violation.
        + status_line(
            "2026-10-23 02:20:00", hint="idle", stale="false", in_flight=0
        ),
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 24, 0, 0),
    )

    violations = metrics["saturated_without_live_work"]
    assert violations["count"] == 3
    assert violations["rate_of_polls"] == 0.5
    assert violations["rate_of_saturated"] == 0.6
    # Busiest minute first, then chronological tie-break.
    assert violations["worst_timestamps"] == [
        {"minute": "2026-10-23 00:00", "polls": 2},
        {"minute": "2026-10-23 02:10", "polls": 1},
    ]
    assert violations["examples"] == [
        "2026-10-23 00:00:02",
        "2026-10-23 00:00:40",
        "2026-10-23 02:10:00",
    ]


def test_saturated_without_available_slots_is_the_recorded_baseline_subset(
    csv, tmp_path
):
    """``saturated`` + ``available_slots=0`` is the recorded 82.3 % derivation."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line(
            "2026-10-23 00:00:01", hint="saturated", available="0", in_flight=1
        )
        + status_line(
            "2026-10-23 00:00:02", hint="saturated", available="0", in_flight=0
        )
        + status_line(
            "2026-10-23 00:00:03", hint="saturated", available="1", in_flight=0
        )
        + status_line("2026-10-23 00:00:04", hint="idle", available="0"),
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 24, 0, 0),
    )

    assert metrics["capacity_hint"]["histogram"] == {"idle": 1, "saturated": 3}
    # Independent of the live-work check: this subset is "the pool reported
    # full", which is what the historical baseline counted.
    assert metrics["saturated_without_available_slots"] == {
        "count": 2,
        "rate_of_polls": 0.5,
    }
    assert metrics["saturated_without_live_work"]["count"] == 2


# ---------------------------------------------------------------------------
# Local dispatches
# ---------------------------------------------------------------------------


def test_local_dispatch_counting_splits_cache_state_and_sessions(csv, tmp_path):
    """Local dispatches are counted once each, split by cache state/session."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        dispatch_line("2026-10-23 00:10:00", session="herdr-1", cache_state="cold")
        + dispatch_line("2026-10-23 00:10:30", session="herdr-1", cache_state="warm")
        + dispatch_line("2026-10-23 00:11:00", session="audit-x", cache_state="warm")
        # Outside the window -> not counted.
        + dispatch_line("2026-10-23 09:00:00", session="herdr-1")
        # A non-dispatch status poll must not be counted as a dispatch.
        + status_line("2026-10-23 00:12:00"),
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert metrics["status_polls"] == 1
    assert metrics["local_dispatches"] == {
        "count": 3,
        "by_cache_state": {"cold": 1, "warm": 2},
        "distinct_sessions": 2,
        "herdr_session_dispatches": 1,
    }


# ---------------------------------------------------------------------------
# herdr downtime dispatch log
# ---------------------------------------------------------------------------


def test_downtime_dispatches_dedupe_utc_conversion_and_window(csv, tmp_path):
    """herdr writes a second enriched row per dispatch; count dispatches once."""
    log = downtime_log(
        tmp_path / "downtime-dispatches.log",
        [
            # Before the window: establishes log coverage, not counted.
            {
                "itemId": "LP-OLD",
                "kind": "plan",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 22, 12, 0)),
                "title": "before",
            },
            # Inside the window, duplicated exactly as herdr appends it.
            {
                "itemId": "LP-A",
                "kind": "intake",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 23, 1, 0)),
                "title": "A",
            },
            {
                "itemId": "LP-A",
                "kind": "intake",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 23, 1, 0)),
                "title": "A",
                "paneId": "wD:p1",
                "enrichment": True,
            },
            {
                "itemId": "LP-B",
                "kind": "implement",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 23, 2, 30)),
                "title": "B",
            },
            # After the window: coverage only.
            {
                "itemId": "LP-LATE",
                "kind": "plan",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 23, 9, 0)),
                "title": "late",
            },
            # Malformed row is skipped, never fatal.
            "not json at all",
        ],
    )

    result = csv.load_downtime_dispatches(
        log,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert result["available"] is True
    assert result["covered"] is True
    assert result["count"] == 2
    assert [entry["itemId"] for entry in result["entries"]] == ["LP-A", "LP-B"]
    assert result["entries"][0]["dispatchedAt"] == "2026-10-23 01:00:00"
    assert result["rows_skipped"] == 1


def test_downtime_dispatches_uncovered_window_reports_zero_not_error(csv, tmp_path):
    """A rolling log that starts after the window cannot prove zero dispatches."""
    log = downtime_log(
        tmp_path / "downtime-dispatches.log",
        [
            {
                "itemId": "LP-A",
                "kind": "plan",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 25, 3, 0)),
                "title": "A",
            }
        ],
    )

    result = csv.load_downtime_dispatches(
        log,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert result["count"] == 0
    assert result["covered"] is False
    assert "not retained" in result["note"]


def test_default_dispatch_log_follows_a_worktree_to_the_shared_worklog(
    csv, tmp_path
):
    """A worktree run must read the main checkout's dispatch log, not its own."""
    main_root = tmp_path / "main"
    worktree = main_root / ".worklog" / "worktrees" / "wl-1"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(
        f"gitdir: {main_root}/.git/worktrees/wl-1\n", encoding="utf-8"
    )

    assert csv._shared_worklog_dir(worktree) == main_root / ".worklog"

    # A normal checkout (``.git`` is a directory) keeps its own worklog.
    plain = tmp_path / "plain"
    (plain / ".git").mkdir(parents=True)
    assert csv._shared_worklog_dir(plain) == plain / ".worklog"


def test_downtime_dispatch_log_missing_is_reported_not_fatal(csv, tmp_path):
    """A missing dispatch log yields a note, not an exception or a wrong zero."""
    result = csv.load_downtime_dispatches(
        tmp_path / "absent.log",
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert result["available"] is False
    assert result["count"] == 0
    assert result["covered"] is False
    assert "not evidence" in result["note"]


# ---------------------------------------------------------------------------
# Discovery, gzip and window boundaries
# ---------------------------------------------------------------------------


def test_dot_dash_and_gz_files_are_all_read_with_per_line_window_filter(
    csv, tmp_path
):
    """Both rotation schemes and ``.gz`` are read; the window is per line.

    A rotated file routinely holds lines past its name-encoded hour, so the
    filter must be the line timestamp — not the file name.
    """
    log_dir = tmp_path / "logs"
    window_start = dt.datetime(2026, 10, 23, 0, 0)
    window_end = dt.datetime(2026, 10, 23, 7, 0)
    # Dot-named plain file: one in-window poll and one after the window.
    write_log(
        log_dir / "proxy.log.2026-10-23_00",
        status_line("2026-10-23 01:00:00")
        + status_line("2026-10-23 08:00:00"),
    )
    # Dash-named gzip file (logrotate safety net).
    write_log(
        log_dir / "proxy.log-2026-10-23_06.gz",
        status_line("2026-10-23 06:00:00", hint="saturated", stale="true"),
    )
    # Dot-named gzip file entirely outside the window.
    write_log(
        log_dir / "proxy.log.2026-10-23_12.gz",
        status_line("2026-10-23 12:00:00"),
    )

    metrics = csv.analyse_status_stream(
        log_dir, since=window_start, until=window_end
    )

    assert metrics["status_polls"] == 2
    assert metrics["slots_stale"]["count"] == 1
    assert metrics["capacity_hint"]["histogram"] == {"idle": 1, "saturated": 1}
    assert metrics["first_seen"] == "2026-10-23 01:00:00"
    assert metrics["last_seen"] == "2026-10-23 06:00:00"


def test_window_bounds_are_inclusive_lower_and_exclusive_upper(csv, tmp_path):
    """Adjacent windows must not double-count a poll on the boundary."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line("2026-10-23 00:00:00")
        + status_line("2026-10-23 06:59:59")
        + status_line("2026-10-23 07:00:00"),
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert metrics["status_polls"] == 2
    assert metrics["first_seen"] == "2026-10-23 00:00:00"
    assert metrics["last_seen"] == "2026-10-23 06:59:59"


# ---------------------------------------------------------------------------
# Coverage verdict
# ---------------------------------------------------------------------------


def _gap_fixture(tmp_path: Path) -> Path:
    """Two files with a 2 h hole inside the requested window."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log.2026-10-23_00",
        status_line("2026-10-23 00:00:10") + status_line("2026-10-23 01:00:00"),
    )
    write_log(
        log_dir / "proxy.log-2026-10-23_03.gz",
        status_line("2026-10-23 03:00:00") + status_line("2026-10-23 06:59:00"),
    )
    return log_dir


def test_gap_inside_window_is_a_coverage_gap_verdict(csv, tmp_path):
    """An interior hole must fail loudly instead of reading as a partial result."""
    log_dir = _gap_fixture(tmp_path)

    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    assert report["coverage"]["covered"] is False
    kinds = [gap["kind"] for gap in report["coverage"]["gaps"]]
    assert kinds == ["between_files"]
    assert report["coverage"]["gaps"][0]["gap_seconds"] == pytest.approx(7200)
    assert report["verdict"]["verdict"] == csv.VERDICT_COVERAGE_GAP


def test_missing_window_head_is_a_coverage_gap(csv, tmp_path):
    """A window that starts before its own data is measuring less than it claims."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line("2026-10-23 02:00:00") + status_line("2026-10-23 06:00:00"),
    )

    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    assert [gap["kind"] for gap in report["coverage"]["gaps"]] == ["window_head"]
    assert report["verdict"]["verdict"] == csv.VERDICT_COVERAGE_GAP


def test_tail_gap_is_informational_until_the_window_is_closed(csv, tmp_path):
    """A live window has no end yet; a closed window with a tail hole has lost data."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line("2026-10-23 00:00:10") + status_line("2026-10-23 02:00:00"),
    )
    bounds = {
        "since": dt.datetime(2026, 10, 23, 0, 0),
        "until": dt.datetime(2026, 10, 23, 7, 0),
        "dispatch_log": None,
    }

    live = csv.measure(log_dir, **bounds)
    assert live["coverage"]["covered"] is True
    assert live["coverage"]["window_end_gap_seconds"] == pytest.approx(18000)
    assert live["verdict"]["verdict"] in {
        csv.VERDICT_PASS,
        csv.VERDICT_TARGETS_MISSED,
    }

    closed = csv.measure(log_dir, window_closed=True, **bounds)
    assert closed["coverage"]["covered"] is False
    assert [gap["kind"] for gap in closed["coverage"]["gaps"]] == ["window_tail"]
    assert closed["verdict"]["verdict"] == csv.VERDICT_COVERAGE_GAP


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_empty_covered_window_reports_zero_and_exits_zero(csv, tmp_path, capsys):
    """An empty window is a result (zeros + verdict), never an error.

    The window is covered (log lines exist inside it) but no status poll was
    written, so every figure is 0 and the verdict says EMPTY — exit 0.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        "2026-10-23 00:00:05,000 - INFO - startup: nothing polled yet\n"
        + "2026-10-23 06:00:00,000 - INFO - still nothing polled\n",
    )

    exit_code = csv.main(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-10-23 00:00",
            "--until",
            "2026-10-23 07:00",
            "--dispatch-log",
            str(tmp_path / "absent.log"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["metrics"]["status_polls"] == 0
    assert payload["metrics"]["slots_stale"]["rate"] == 0.0
    assert payload["verdict"]["verdict"] == csv.VERDICT_EMPTY


def test_window_outside_retained_logs_is_a_coverage_gap(csv, tmp_path, capsys):
    """A window the retained logs never reach is a gap, not an empty result."""
    log_dir = tmp_path / "logs"
    write_log(log_dir / "proxy.log", status_line("2026-10-23 12:00:00"))

    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    assert report["coverage"]["covered"] is False
    assert [gap["kind"] for gap in report["coverage"]["gaps"]] == [
        "outside_retention"
    ]
    assert report["verdict"]["verdict"] == csv.VERDICT_COVERAGE_GAP

    exit_code = csv.main(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-10-23 00:00",
            "--until",
            "2026-10-23 07:00",
            "--dispatch-log",
            str(tmp_path / "absent.log"),
            "--json",
        ]
    )
    assert exit_code == 3
    capsys.readouterr()


def test_cli_exits_three_on_coverage_gap(csv, tmp_path, capsys):
    log_dir = _gap_fixture(tmp_path)

    exit_code = csv.main(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-10-23 00:00",
            "--until",
            "2026-10-23 07:00",
            "--dispatch-log",
            str(tmp_path / "absent.log"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["verdict"]["verdict"] == csv.VERDICT_COVERAGE_GAP


def test_json_document_emits_every_required_figure(csv, tmp_path, capsys):
    """The ``--json`` document carries each AC1 figure and the derivation meta."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line(
            "2026-10-23 00:00:01", hint="saturated", stale="true", available="0"
        )
        + status_line("2026-10-23 00:00:02", hint="idle", stale="false")
        + dispatch_line("2026-10-23 00:00:03", session="herdr-9"),
    )
    dispatch_log = downtime_log(
        tmp_path / "downtime-dispatches.log",
        [
            {
                "itemId": "LP-A",
                "kind": "implement",
                "dispatchedAt": utc_z(dt.datetime(2026, 10, 23, 1, 0)),
                "title": "A",
            }
        ],
    )

    exit_code = csv.main(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-10-23 00:00",
            "--until",
            "2026-10-23 07:00",
            "--dispatch-log",
            str(dispatch_log),
            "--label",
            "unit-test",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["meta"]["tool"] == "scripts/capacity_signal_verification.py"
    assert payload["meta"]["work_item"] == "LP-0MVOG6YRA000T9PK"
    assert payload["meta"]["label"] == "unit-test"
    assert payload["meta"]["files"] == ["proxy.log"]

    metrics = payload["metrics"]
    assert metrics["status_polls"] == 2
    assert metrics["slots_stale"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert metrics["capacity_hint"]["histogram"] == {"idle": 1, "saturated": 1}
    assert metrics["saturated_without_live_work"]["count"] == 1
    assert metrics["saturated_without_available_slots"]["count"] == 1
    assert metrics["local_dispatches"]["count"] == 1

    assert payload["herdr_downtime_dispatches"]["count"] == 1
    assert payload["coverage"]["covered"] is True
    assert payload["verdict"]["verdict"] in {
        csv.VERDICT_PASS,
        csv.VERDICT_TARGETS_MISSED,
    }
    # The single violating poll is quoted so it can be found in the log.
    assert metrics["saturated_without_live_work"]["examples"] == [
        "2026-10-23 00:00:01"
    ]


def test_cli_rejects_missing_log_directory(csv, tmp_path, capsys):
    exit_code = csv.main(["--log-dir", str(tmp_path / "nope"), "--json"])

    assert exit_code == 1
    assert "log directory not found" in capsys.readouterr().err


def test_fail_on_target_miss_exits_two(csv, tmp_path, capsys):
    """Targets are advisory by default; a CI caller can make them fatal."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line(
            "2026-10-23 00:00:01", hint="saturated", stale="true", available="0"
        ),
    )
    argv = [
        "--log-dir",
        str(log_dir),
        "--since",
        "2026-10-23 00:00",
        "--until",
        "2026-10-23 07:00",
        "--dispatch-log",
        str(tmp_path / "absent.log"),
        "--json",
    ]

    assert csv.main(argv) == 0
    assert csv.main([*argv, "--fail-on-target-miss"]) == 2
    capsys.readouterr()
