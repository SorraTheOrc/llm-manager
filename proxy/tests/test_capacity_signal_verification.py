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
    detail_stale: str | None = None,
    latency_ms: int = 3,
) -> str:
    """One ``status_request`` log line, as the proxy writes it.

    ``detail_stale`` emits the per-source freshness flag the v3 status surface
    publishes next to the legacy ``slots_stale`` union; ``latency_ms`` is the
    handler's own response latency, the evidence that separates a fast fresh
    poll from one that exhausted the ``/slots`` query budget.
    """
    fields = [
        f"active_query={active}",
        f"available_slots={available}",
        f"capacity_hint={hint}",
        "current_model=Qwen3",
        f"latency_ms={latency_ms}",
        "llama_server_running=true",
        f"local_active_query={local_active}",
    ]
    if in_flight is not None:
        fields.append(f"local_in_flight_requests={in_flight}")
    fields += [
        "model_switch_in_progress=false",
        f"slots_stale={stale}",
    ]
    if detail_stale is not None:
        fields.append(f"slots_detail_stale={detail_stale}")
    fields += [
        f"status_schema_version={schema}",
        "total_slots=2",
    ]
    return f"{ts},000 - INFO - status_request " + " ".join(fields) + "\n"


def lease_released_line(
    ts: str, *, reason: str = "prefill_deadline", session: str = "s1"
) -> str:
    """A local dispatch lease release, the proxy's abandonment record."""
    return (
        f"{ts},000 - INFO - lease_released session={session} "
        f"reason={reason} endpoint=http://localhost:8080\n"
    )


def restart_line(ts: str) -> str:
    """The marker the proxy writes when it rotates the log aside on restart."""
    return (
        f"{ts},000 - INFO - Rotated previous proxy log aside: "
        "proxy.log.2026-10-24_173748 (pre-restart window preserved)\n"
    )


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


# ---------------------------------------------------------------------------
# Attribution: stale polls by cache source, violating polls by cause
# ---------------------------------------------------------------------------


def test_stale_polls_attributed_to_cache_source_and_latency_band(csv, tmp_path):
    """Every stale poll is attributed to a served-from-cache source.

    ``slots_stale`` is the union of the counts cache (LP-0MSVP7XJ6008QPKX) and
    the per-slot detail cache (LP-0MUFSVXID0039ZAQ); a poll that carries the
    union flag but neither per-source flag (a pre-v3 schema) cannot be
    attributed and must be counted as such rather than silently folded in.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line(
            "2026-10-23 00:00:01",
            hint="idle",
            stale="true",
            detail_stale="true",
            latency_ms=3100,
        )
        # Fresh and fast: no upstream /slots budget was exhausted.
        + status_line(
            "2026-10-23 00:00:02",
            hint="idle",
            stale="false",
            detail_stale="false",
            latency_ms=4,
        )
        # Counts cached (detail fresh): the counts query timed out.
        + status_line(
            "2026-10-23 00:00:03",
            hint="saturated",
            stale="true",
            detail_stale="false",
            in_flight=1,
            latency_ms=2400,
        )
        # Union flag only, and no latency field: no per-source and no latency
        # evidence to attribute it with.
        + "2026-10-23 00:00:04,000 - INFO - status_request "
        "capacity_hint=idle slots_stale=true status_schema_version=3 "
        "total_slots=2\n"
        + status_line(
            "2026-10-23 00:00:05",
            hint="idle",
            stale="true",
            detail_stale="true",
            latency_ms=6100,
        ),
    )

    metrics = csv.analyse_status_stream(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 24, 0, 0),
    )

    attribution = metrics["slots_stale_attribution"]
    assert attribution["by_source"] == {
        "detail_cache": 2,
        "counts_cache": 1,
        "unknown": 1,
    }
    assert attribution["by_latency_band"] == {
        "lt_1s": 0,
        "1s_to_2s": 0,
        "2s_to_5s": 2,
        "gte_5s": 1,
        "unknown": 1,
    }
    assert attribution["unattributed"] == 1


def test_abandonment_events_are_local_lease_releases_and_restarts(csv, tmp_path):
    """Only local abandonments are evidence; an idle timeout is not.

    A ``lease_released`` with ``reason=prefill_deadline``/``session_evicted``/
    ``no_progress_watchdog`` records that the proxy stopped tracking a local
    dispatch while llama-server may still be finishing it. ``idle_timeout``
    is the normal end of a completed stream and proves nothing about a slot
    that a later ``/slots`` snapshot still reports as processing.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line(
            "2026-10-23 00:10:00", reason="prefill_deadline", session="s1"
        )
        + lease_released_line(
            "2026-10-23 00:20:00", reason="idle_timeout", session="s2"
        )
        + lease_released_line(
            "2026-10-23 00:30:00", reason="session_evicted", session="s3"
        )
        + restart_line("2026-10-23 00:40:00")
        # Outside the window -> not evidence for it.
        + lease_released_line(
            "2026-10-23 09:00:00", reason="prefill_deadline", session="s4"
        ),
    )

    events = csv.analyse_abandonment_events(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert [(event["timestamp"], event["kind"], event["reason"]) for event in events] == [
        ("2026-10-23 00:10:00", "lease_released", "prefill_deadline"),
        ("2026-10-23 00:30:00", "lease_released", "session_evicted"),
        ("2026-10-23 00:40:00", "proxy_restart", None),
    ]


def test_violations_attributed_to_the_preceding_abandonment(csv):
    """Each violating poll is attributed to the nearest preceding event.

    A poll takes the most recent abandonment inside the lookback window and is
    reported as unattributed (never guessed) when there is none — including a
    violation whose only nearby event is *later* than the poll.
    """
    violations = [
        {
            "timestamp": "2026-10-23 00:11:00",
            "latency_ms": 3100,
            "slots_detail_stale": False,
            "available_slots": 0,
        },
        {
            "timestamp": "2026-10-23 00:12:30",
            "latency_ms": 4200,
            "slots_detail_stale": False,
            "available_slots": 0,
        },
        {
            "timestamp": "2026-10-23 00:30:00",
            "latency_ms": 900,
            "slots_detail_stale": False,
            "available_slots": 0,
        },
    ]
    events = [
        {
            "timestamp": "2026-10-23 00:10:00",
            "kind": "lease_released",
            "reason": "prefill_deadline",
            "session": "s1",
        },
        {
            "timestamp": "2026-10-23 01:00:00",
            "kind": "lease_released",
            "reason": "session_evicted",
            "session": "s2",
        },
    ]

    attribution = csv.attribute_violations(
        violations, events, lookback_seconds=600
    )

    assert attribution["attributed_polls"] == 2
    assert attribution["unattributed_polls"] == 1
    assert attribution["by_cause"] == {"prefill_deadline": 2}
    assert attribution["unattributed_timestamps"] == ["2026-10-23 00:30:00"]
    assert attribution["rows"][0]["cause"] == "prefill_deadline"
    assert attribution["rows"][0]["evidence_timestamp"] == "2026-10-23 00:10:00"
    assert attribution["rows"][0]["seconds_since_evidence"] == 60
    # The 01:00 event is *after* the 00:30 poll, so it cannot explain it.
    assert attribution["rows"][2]["cause"] is None


def test_measure_joins_abandonment_evidence_into_the_report(csv, tmp_path):
    """``measure`` carries the attribution and the evidence it was built on."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line("2026-10-23 00:10:00", reason="prefill_deadline")
        + status_line(
            "2026-10-23 00:11:00",
            hint="saturated",
            stale="false",
            detail_stale="false",
            in_flight=0,
            available="0",
            latency_ms=3100,
        ),
    )

    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    assert report["metrics"]["saturated_without_live_work"]["count"] == 1
    attribution = report["attribution"]
    assert attribution["attributed_polls"] == 1
    assert attribution["unattributed_polls"] == 0
    assert attribution["by_cause"] == {"prefill_deadline": 1}
    assert report["abandonment_events"]["count"] == 1
    assert report["abandonment_events"]["by_reason"] == {"prefill_deadline": 1}
    # The verdict still reports the violation; attribution explains it, it does
    # not excuse it.
    missed = [
        check for check in report["verdict"]["checks"] if not check["pass"]
    ]
    assert any("saturated" in check["target"] for check in missed)
    assert any("attributed" in note for note in report["verdict"]["notes"])


def test_write_artifacts_emits_json_text_and_attribution_csv(csv, tmp_path):
    """One command persists the raw measurement and the per-poll attribution."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line("2026-10-23 00:10:00", reason="session_evicted")
        + status_line(
            "2026-10-23 00:11:00",
            hint="saturated",
            stale="true",
            detail_stale="false",
            in_flight=0,
            available="0",
            latency_ms=3100,
        )
        + status_line("2026-10-23 00:12:00", hint="idle", stale="true",
                      detail_stale="true", latency_ms=5100),
    )
    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    out_dir = tmp_path / "artifacts"
    written = csv.write_artifacts(report, out_dir)

    # LP-0MVRMI5HK00558TB AC2: the cooldown-skip CSV is an ADDITIONAL
    # artifact; the pre-existing three keep their names and contents.
    assert sorted(path.name for path in written) == [
        "measurement.json",
        "measurement.txt",
        "prefill_thrash_skips.csv",
        "saturated_without_live_work_attribution.csv",
    ]
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "measurement.json",
        "measurement.txt",
        "prefill_thrash_skips.csv",
        "saturated_without_live_work_attribution.csv",
    ]
    payload = json.loads((out_dir / "measurement.json").read_text())
    assert payload["attribution"]["by_cause"] == {"session_evicted": 1}
    assert "VERDICT" in (out_dir / "measurement.txt").read_text()

    rows = (out_dir / "saturated_without_live_work_attribution.csv").read_text(
    ).splitlines()
    assert rows[0] == (
        "timestamp,cause,evidence_timestamp,seconds_since_evidence,"
        "latency_ms,slots_detail_stale,available_slots"
    )
    assert rows[1].startswith("2026-10-23 00:11:00,session_evicted,"
                             "2026-10-23 00:10:00,60,")


def test_cli_write_artifacts_and_lookback(csv, tmp_path, capsys):
    """The CLI persists the same artifacts and honours the lookback bound."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line("2026-10-23 00:10:00", reason="prefill_deadline")
        + status_line(
            "2026-10-23 00:30:00",
            hint="saturated",
            stale="true",
            detail_stale="false",
            in_flight=0,
            available="0",
        ),
    )
    out_dir = tmp_path / "artifacts"
    argv = [
        "--log-dir",
        str(log_dir),
        "--since",
        "2026-10-23 00:00",
        "--until",
        "2026-10-23 07:00",
        "--dispatch-log",
        str(tmp_path / "absent.log"),
        "--write-artifacts",
        str(out_dir),
        "--json",
    ]

    assert csv.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    # 20 minutes after the abandonment, inside the 600 s default lookback?
    # No: 1200 s > 600 s, so it must be reported unattributed.
    assert payload["attribution"]["unattributed_polls"] == 1
    assert (out_dir / "measurement.json").is_file()

    assert csv.main([*argv, "--attribution-lookback-seconds", "1800"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attribution"]["by_cause"] == {"prefill_deadline": 1}


def test_window_lead_in_abandonment_attributes_the_first_poll(csv, tmp_path):
    """The lookback may read *before* the window; the tally may not.

    The first polls of a window are routinely explained by an abandonment
    moments before it, so the attribution scan starts one lookback earlier —
    but the reported abandonment tally stays scoped to the measured window.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line("2026-10-22 23:58:00", reason="prefill_deadline")
        + status_line(
            "2026-10-23 00:00:20",
            hint="saturated",
            stale="false",
            detail_stale="false",
            in_flight=0,
            available="0",
        ),
    )

    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
        dispatch_log=None,
    )

    assert report["attribution"]["attributed_polls"] == 1
    assert report["attribution"]["rows"][0]["seconds_since_evidence"] == 140
    assert report["abandonment_events"]["count"] == 0


def test_bulk_eviction_keeps_one_event_per_session(csv, tmp_path):
    """A bulk eviction releases several leases in the same second.

    Each is a real abandonment (a distinct session lost its slot), so the tally
    must count them all; only a genuinely identical line is de-duplicated.
    """
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        lease_released_line("2026-10-23 00:10:00", reason="session_evicted",
                            session="s1")
        + lease_released_line("2026-10-23 00:10:00", reason="session_evicted",
                              session="s2")
        + lease_released_line("2026-10-23 00:10:00", reason="session_evicted",
                              session="s3"),
    )

    events = csv.analyse_abandonment_events(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 7, 0),
    )

    assert [event["session"] for event in events] == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Post-deadline prefill-thrash cooldown skips (LP-0MVT2KVVS002YZT8)
#
# Parent LP-0MVRMI5HK00558TB AC2 needs the before/after numbers to come from the
# same tool invocation shape as everything else; the 916-skip / 336-free-GPU
# baseline was derived with ad-hoc greps. The block is strictly additive: every
# existing metric keeps its name, definition and JSON shape.
# ---------------------------------------------------------------------------


def cooldown_skip_line(
    ts: str,
    *,
    session: str = "s1",
    provider: str = "local-llama",
    preference: str = "prefer-local",
    cooldown_seconds: int = 540,
) -> str:
    """One ``routing_skip_local reason=prefill_thrash_cooldown`` log line."""
    return (
        f"{ts},000 - INFO - routing_skip_local provider={provider} "
        f"model=Qwen3 session={session} preference={preference} "
        f"cooldown_remaining={cooldown_seconds}s "
        "reason=prefill_thrash_cooldown → skipping local, routing to next "
        "remote provider\n"
    )


def cooldown_released_line(ts: str, *, session: str = "s1", reason: str = "free_gpu") -> str:
    """The release line a free-GPU (or other) release writes."""
    return (
        f"{ts},000 - INFO - prefill_thrash_cooldown_released session={session} "
        f"reason={reason} cooldown_remaining=300s endpoint=default\n"
    )


WINDOW = {
    "since": dt.datetime(2026, 10, 24, 20, 0),
    "until": dt.datetime(2026, 10, 25, 7, 0),
}


def test_prefill_thrash_skips_counted_with_herdr_split(csv, tmp_path):
    """Total skips, distinct sessions, and the ``herdr-*`` subset."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        cooldown_skip_line("2026-10-24 23:39:17", session="herdr-a")
        + cooldown_skip_line("2026-10-24 23:40:17", session="herdr-a")
        + cooldown_skip_line("2026-10-25 00:00:49", session="herdr-b")
        + cooldown_skip_line("2026-10-25 01:00:00", session="agent-x"),
    )

    thrash = csv.analyse_prefill_thrash_skips(log_dir, **WINDOW)

    assert thrash["count"] == 4
    assert thrash["distinct_sessions"] == 3
    assert thrash["herdr_skips"] == 3
    assert thrash["herdr_sessions"] == 2
    assert thrash["sessions"][0] == {"session": "herdr-a", "skips": 2}
    # The remaining fields are read off the line, not guessed.
    assert thrash["rows"][0]["provider"] == "local-llama"
    assert thrash["rows"][0]["cooldown_remaining_seconds"] == 540


def test_prefill_thrash_skip_free_gpu_attribution(csv, tmp_path):
    """The free-GPU rule: nearest preceding poll within 120 s must be fresh
    and show no live local work; the ``idle`` hint is a sub-count."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        # skip 1: nearest poll 5 s earlier, fresh, nothing in flight, idle
        status_line(
            "2026-10-24 23:39:12", hint="idle", detail_stale="false", in_flight=0
        )
        + cooldown_skip_line("2026-10-24 23:39:17", session="herdr-a")
        # skip 2: nearest poll 10 s earlier, fresh, nothing in flight, but
        # abandoned_busy (not idle) -> counts as free, not as idle-hint
        + status_line(
            "2026-10-24 23:40:10",
            hint="abandoned_busy",
            detail_stale="false",
            in_flight=0,
        )
        + cooldown_skip_line("2026-10-24 23:40:20", session="herdr-a")
        # skip 3: nearest poll shows live work -> NOT a free-GPU skip
        + status_line(
            "2026-10-24 23:41:00", hint="saturated", detail_stale="false", in_flight=1
        )
        + cooldown_skip_line("2026-10-24 23:41:05", session="agent-x")
        # skip 4: nearest poll is stale detail -> NOT a free-GPU skip
        + status_line(
            "2026-10-24 23:42:00", hint="idle", detail_stale="true", in_flight=0
        )
        + cooldown_skip_line("2026-10-24 23:42:05", session="agent-y")
        # skip 5: nearest poll is 130 s older than the lookback -> NOT counted
        + status_line(
            "2026-10-24 23:44:00", hint="idle", detail_stale="false", in_flight=0
        )
        + cooldown_skip_line("2026-10-24 23:46:10", session="agent-z"),
    )

    thrash = csv.analyse_prefill_thrash_skips(log_dir, **WINDOW)

    assert thrash["count"] == 5
    # Strict rule (parent Appendix): fresh poll AND nothing in flight.
    assert thrash["skips_with_free_gpu"] == 2
    assert thrash["skips_with_free_gpu_idle_hint"] == 1
    # Relaxed rule (the population the recorded 336 baseline came from): the
    # stale-detail poll is included, so three skips have no live local work.
    assert thrash["skips_without_live_local_work"] == 3
    assert thrash["skips_without_live_local_work_idle_hint"] == 2
    assert [row["free_gpu"] for row in thrash["rows"]] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert [row["no_live_local_work"] for row in thrash["rows"]] == [
        True,
        True,
        False,
        True,
        False,
    ]
    # Both rules are stated in the report so the numbers are re-derivable.
    assert "local_in_flight_requests=0" in thrash["attribution_rule"]
    assert "slots_detail_stale=false" in thrash["attribution_rule"]
    assert "relaxed" in thrash["attribution_rule"]
    assert "120s" in thrash["attribution_rule"]
    assert thrash["free_gpu_lookback_seconds"] == 120.0


def test_prefill_thrash_skip_before_any_poll_is_not_attributed(csv, tmp_path):
    """A skip with no preceding poll (head of the window) is never guessed."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        cooldown_skip_line("2026-10-24 20:00:01", session="herdr-a")
        + status_line(
            "2026-10-24 20:00:30", hint="idle", detail_stale="false", in_flight=0
        ),
    )

    thrash = csv.analyse_prefill_thrash_skips(log_dir, **WINDOW)

    assert thrash["count"] == 1
    assert thrash["skips_with_free_gpu"] == 0


def test_prefill_thrash_releases_counted_by_reason(csv, tmp_path):
    """Releases are counted by reason; an older window with no release line
    reads zero releases rather than failing."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        cooldown_skip_line("2026-10-24 23:00:00", session="s1")
        + cooldown_released_line("2026-10-24 23:01:00", reason="free_gpu")
        + cooldown_released_line("2026-10-24 23:02:00", reason="free_gpu")
        + cooldown_released_line("2026-10-24 23:03:00", reason="comparable_success"),
    )

    thrash = csv.analyse_prefill_thrash_skips(log_dir, **WINDOW)

    assert thrash["releases"] == {
        "count": 3,
        "by_reason": {"comparable_success": 1, "free_gpu": 2},
    }

    # Older logs: no release line at all -> zero, not an error.
    bare = tmp_path / "bare"
    write_log(bare / "proxy.log", cooldown_skip_line("2026-10-24 23:00:00"))
    assert csv.analyse_prefill_thrash_skips(bare, **WINDOW)["releases"] == {
        "count": 0,
        "by_reason": {},
    }


def test_prefill_thrash_skips_window_and_rotation(csv, tmp_path):
    """Same per-line window semantics as every other metric, and gzip/rotated
    files are read (the file name never bounds its content)."""
    log_dir = tmp_path / "logs"
    # In-window lines inside a .gz whose name says another hour, plus an
    # out-of-window line in a plain file.
    write_log(
        log_dir / "proxy.log-2026-10-24_23.gz",
        cooldown_skip_line("2026-10-24 23:30:00", session="herdr-a")
        + cooldown_skip_line("2026-10-25 07:00:00", session="herdr-a"),
    )
    write_log(
        log_dir / "proxy.log-2026-10-25_00",
        cooldown_skip_line("2026-10-25 00:10:00", session="herdr-b")
        + cooldown_skip_line("2026-10-24 19:59:59", session="herdr-b"),
    )

    thrash = csv.analyse_prefill_thrash_skips(log_dir, **WINDOW)

    assert thrash["count"] == 2
    assert thrash["distinct_sessions"] == 2


def test_prefill_thrash_block_is_additive_and_carries_coverage(csv, tmp_path):
    """The block is a new key; existing metrics keep their shape, and a
    coverage gap is carried into the block instead of reading as zero skips."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line("2026-10-23 00:00:01", hint="idle", detail_stale="false", in_flight=0)
        + dispatch_line("2026-10-23 00:00:02", session="s1")
        + cooldown_skip_line("2026-10-23 00:00:03", session="herdr-a"),
    )

    # The window starts a day before the retained data: a hard head gap, so
    # the coverage verdict must say the window is not fully covered.
    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 22, 0, 0),
        until=dt.datetime(2026, 10, 23, 6, 0),
        dispatch_log=None,
    )

    # Existing shapes untouched.
    assert report["metrics"]["status_polls"] == 1
    assert report["metrics"]["local_dispatches"]["count"] == 1
    assert report["metrics"]["saturated_without_live_work"]["count"] == 0
    assert set(report["abandonment_events"]) == {"count", "by_reason"}

    thrash = report["prefill_thrash_cooldown"]
    assert thrash["count"] == 1
    assert thrash["window_covered"] is False
    assert thrash["coverage_gaps"] >= 1
    assert "not a complete figure" in csv.render(report)


def test_prefill_thrash_csv_artifact_is_additional(csv, tmp_path):
    """The per-skip CSV is written alongside the existing artifacts."""
    log_dir = tmp_path / "logs"
    write_log(
        log_dir / "proxy.log",
        status_line(
            "2026-10-23 00:00:10", hint="idle", detail_stale="false", in_flight=0
        )
        + cooldown_skip_line("2026-10-23 00:00:20", session="herdr-a"),
    )
    report = csv.measure(
        log_dir,
        since=dt.datetime(2026, 10, 23, 0, 0),
        until=dt.datetime(2026, 10, 23, 0, 30),
        dispatch_log=None,
    )

    out_dir = tmp_path / "artifacts"
    csv.write_artifacts(report, out_dir)

    rows = (out_dir / "prefill_thrash_skips.csv").read_text().splitlines()
    assert rows[0] == (
        "timestamp,session,provider,preference,cooldown_remaining_seconds,"
        "herdr,free_gpu,free_gpu_capacity_hint,free_gpu_poll_timestamp,"
        "no_live_local_work,no_live_local_work_capacity_hint,"
        "nearest_poll_timestamp"
    )
    assert len(rows) == 2
    assert "herdr-a" in rows[1]
    assert ",True,True,idle," in rows[1]
