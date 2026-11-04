"""Attribution of ``prefill_deadline`` evictions to their admission path.

LP-0MVZ5WV6W008UO5F AC1. The report counts
``local_eviction_after_acceptance reason=prefill_deadline`` events, but the
count alone cannot say *why* an accepted dispatch was allowed to start: the
free-gate rung's accepted trade, an override that never consulted the guard,
a ``high`` priority dispatch the guard is inert for, or an uncalibrated bucket
are different defects with different fixes.

These tests pin the attribution rules of
``scripts/attribute_prefill_deadline.py`` against synthetic log lines shaped
exactly like the proxy's real ones (see the field contract in the skill's
``SKILL.md`` triage rule), including rotation (gzip) handling and the window
filter.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import attribute_prefill_deadline as attribution  # noqa: E402

WINDOW_START = datetime(2026, 11, 1, 11, 0, 0)
WINDOW_END = datetime(2026, 11, 2, 11, 0, 0)


def _ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S") + ",000"


def _eviction(
    ts: datetime,
    session: str,
    *,
    new_tokens: int = 12000,
    estimated_tokens: int = 78000,
    eta_prefill_s: str = "87.0",
    elapsed_s: float = 120.5,
    reason: str = "prefill_deadline",
) -> str:
    return (
        f"{_ts(ts)} - ERROR - local_eviction_after_acceptance session={session} "
        f"reason={reason} estimated_tokens={estimated_tokens} "
        f"new_tokens={new_tokens} eta_prefill_s={eta_prefill_s} "
        f"elapsed_s={elapsed_s} slot_id=0 llama_task_id=unknown "
        f"endpoint=http://localhost:8080"
    )


def _guard(
    ts: datetime,
    session: str,
    *,
    decision: str = "admit",
    reason: str = "eta_exceeds_budget_direct",
    eta_prefill_s: float = 87.0,
    eta_occupancy_s: float = 0.0,
    projected_wait_s: float = 87.0,
    budget_s: float = 30.0,
    bound_s: float = 114.0,
    deadline_budget_s: float = 114.0,
    new_tokens: int = 12000,
    busy_slots: int = 1,
    gate_occupancy: int = 0,
    priority: str = "medium",
    free_slot: bool = True,
) -> str:
    return (
        f"{_ts(ts)} - INFO - local_admission_guard decision={decision} "
        f"reason={reason} enforced=false eta_prefill_s={eta_prefill_s:.1f} "
        f"eta_occupancy_s={eta_occupancy_s:.1f} "
        f"projected_wait_s={projected_wait_s:.1f} budget_s={budget_s:.1f} "
        f"bound_s={bound_s:.1f} deadline_s=120.0 "
        f"deadline_budget_s={deadline_budget_s:.1f} new_tokens={new_tokens} "
        f"budget_tokens=25000 rate_bucket=solitary contended_samples=0 "
        f"busy_slots={busy_slots} in_flight_prefills={gate_occupancy} "
        f"gate_occupancy={gate_occupancy} free_slot={str(free_slot).lower()} "
        f"priority={priority} policy=reject model=Qwen3 samples=5 "
        f"provider=local-qwen3 session={session}"
    )


def _inert_no_budget(ts: datetime, session: str, *, priority: str = "high") -> str:
    """A logged inert ``no_budget`` decision (every time bound disabled).

    Inert decisions log nothing by default, so this line is only produced when
    an inert decision was explicitly logged; it is what separates a genuine
    ``high_priority_inert`` from an unlogged eviction (LP-0MVZQUT3A0039MLY).
    """
    return (
        f"{_ts(ts)} - INFO - local_admission_guard decision=admit "
        f"reason=no_budget enforced=false eta_prefill_s=0.0 "
        f"eta_occupancy_s=0.0 projected_wait_s=0.0 budget_s=0.0 bound_s=0.0 "
        f"deadline_s=120.0 deadline_budget_s=114.0 new_tokens=12000 "
        f"budget_tokens=25000 rate_bucket=solitary contended_samples=0 "
        f"busy_slots=0 in_flight_prefills=0 gate_occupancy=0 free_slot=true "
        f"priority={priority} policy=reject model=Qwen3 samples=5 "
        f"provider=local-qwen3 session={session}"
    )


def _priority(ts: datetime, session: str, priority: str) -> str:
    return (
        f"{_ts(ts)} - INFO - priority_routing model=unknown priority={priority} "
        f"providers=7 preference=remote-ok session={session}"
    )


def _idle_admit(ts: datetime, session: str, new_tokens: int = 12000) -> str:
    return (
        f"{_ts(ts)} - INFO - routing_idle_admit provider=local-qwen3 model=Qwen3 "
        f"reason=economic_bypass_idle idle_seconds=61.2 threshold_seconds=60.0 "
        f"estimated_tokens=78000 new_tokens={new_tokens} cached_ratio=0.82 "
        f"cache_state=warm session={session}"
    )


def _write_log(directory: Path, name: str, lines: list[str], *, compress: bool = False) -> Path:
    path = directory / name
    payload = "\n".join(lines) + "\n"
    if compress:
        with gzip.open(path, "wt") as handle:
            handle.write(payload)
    else:
        path.write_text(payload)
    return path


def _attribute(directory: Path, **kwargs) -> dict:
    return attribution.attribute(
        str(directory), WINDOW_START, WINDOW_END,
        include_rows=kwargs.get("include_rows", True),
    )


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    """One eviction per admission path, each joined at its dispatch start."""
    moment = datetime(2026, 11, 1, 22, 0, 0)
    lines: list[str] = []

    # Rung-accepted: priced 87 s — above the 30 s queue budget, inside the
    # 114 s deadline bound, so the free-gate rung admitted it.
    rung = "sess-rung"
    lines += [
        _priority(moment, rung, "medium"),
        _guard(moment, rung, projected_wait_s=87.0),
        _eviction(moment + timedelta(seconds=120), rung),
    ]

    # Priced inside the queue budget yet stalled the full deadline.
    stalled = "sess-stalled"
    lines += [
        _guard(moment, stalled, projected_wait_s=5.3, eta_prefill_s=5.3,
               new_tokens=73, bound_s=114.0),
        _eviction(moment + timedelta(seconds=121), stalled, new_tokens=73,
                  eta_prefill_s="5.3", elapsed_s=121.0),
    ]

    # Idle-capacity override: the guard never adjudicated.
    idle = "sess-idle"
    lines += [
        _priority(moment, idle, "medium"),
        _idle_admit(moment, idle, new_tokens=14041),
        _eviction(moment + timedelta(seconds=120), idle, new_tokens=14041),
    ]

    # ``high`` priority with a genuine inert ``no_budget`` decision: the only
    # remaining shape that still attributes to ``high_priority_inert``.
    high = "sess-high"
    lines += [
        _priority(moment, high, "high"),
        _inert_no_budget(moment, high),
        _eviction(moment + timedelta(seconds=121), high, new_tokens=20723,
                  eta_prefill_s="278.8", elapsed_s=121.0),
    ]

    # An uncalibrated bucket: nothing is logged at admission either.
    inert = "sess-inert"
    lines += [
        _priority(moment, inert, "medium"),
        _eviction(moment + timedelta(seconds=120), inert, new_tokens=14041),
    ]

    # Non-deadline reasons must not be attributed.
    lines.append(
        _eviction(moment + timedelta(seconds=5), "sess-other",
                  reason="measured_cache_deficit", elapsed_s=1.0)
    )
    # Out-of-window noise, including a line that looks like a guard admit.
    outside = datetime(2026, 10, 20, 9, 0, 0)
    lines += [
        _guard(outside, "sess-old"),
        _eviction(outside + timedelta(seconds=120), "sess-old"),
    ]

    _write_log(tmp_path, "proxy.log", lines)
    return tmp_path


def test_each_admission_path_is_attributed_exactly_once(log_dir: Path) -> None:
    summary = _attribute(log_dir)

    assert summary["evictions"] == 5
    assert summary["by_admission"] == {
        attribution.RUNG_ACCEPTED: 1,
        attribution.PRICED_WITHIN_QUEUE_BUDGET: 1,
        "idle_override": 1,
        attribution.HIGH_PRIORITY_INERT: 1,
        attribution.INERT_UNLOGGED: 1,
    }


def test_rung_accepted_row_carries_the_priced_wait_and_bound(log_dir: Path) -> None:
    rows = {row["session"]: row for row in _attribute(log_dir)["rows"]}

    rung = rows["sess-rung"]
    assert rung["admission"] == attribution.RUNG_ACCEPTED
    assert rung["guard_projected_s"] == pytest.approx(87.0)
    assert rung["guard_budget_s"] == pytest.approx(30.0)
    assert rung["guard_bound_s"] == pytest.approx(114.0)
    assert rung["priced_eta_s"] == pytest.approx(87.0)
    assert rung["elapsed_s"] == pytest.approx(120.5)


def test_dispatch_start_join_ignores_a_stale_decision(log_dir: Path) -> None:
    """A guard line from an earlier attempt must not be borrowed."""
    moment = datetime(2026, 11, 1, 23, 0, 0)
    session = "sess-stale"
    stale_guard = _guard(moment - timedelta(seconds=540), session,
                         decision="reject", projected_wait_s=117.2,
                         bound_s=30.0)
    _write_log(
        log_dir, "proxy.log.2026-11-01_21",
        [
            stale_guard,
            _priority(moment, session, "medium"),
            _eviction(moment + timedelta(seconds=120), session),
        ],
    )

    row = next(
        r for r in _attribute(log_dir)["rows"] if r["session"] == session
    )
    assert row["admission"] == attribution.INERT_UNLOGGED


def test_gzipped_rotation_files_inside_the_window_are_scanned(tmp_path: Path) -> None:
    moment = datetime(2026, 11, 1, 12, 0, 0)
    session = "sess-gz"
    _write_log(
        tmp_path, "proxy.log.2026-11-01_13.gz",
        [
            _priority(moment, session, "high"),
            _inert_no_budget(moment, session),
            _eviction(moment + timedelta(seconds=120), session, new_tokens=23027),
        ],
        compress=True,
    )

    summary = _attribute(tmp_path)
    assert summary["by_admission"] == {attribution.HIGH_PRIORITY_INERT: 1}
    assert summary["max_new_tokens"] == 23027


def test_unlogged_high_priority_eviction_is_inert_unlogged(tmp_path: Path) -> None:
    """A high-priority eviction with no guard line is unlogged, not inert-high.

    LP-0MVZQUT3A0039MLY AC3: after the guard fix ``high`` is bounded like every
    other priority, so an eviction with no ``no_budget`` line is as
    un-attributable as any other unlogged inert decision.
    """
    moment = datetime(2026, 11, 1, 22, 0, 0)
    session = "sess-high-unlogged"
    _write_log(
        tmp_path, "proxy.log",
        [
            _priority(moment, session, "high"),
            _eviction(moment + timedelta(seconds=121), session, new_tokens=20723,
                      eta_prefill_s="278.8", elapsed_s=121.0),
        ],
    )

    summary = _attribute(tmp_path)
    assert summary["by_admission"] == {attribution.INERT_UNLOGGED: 1}
    row = summary["rows"][0]
    assert row["priority"] == "high"
    assert row["admission"] == attribution.INERT_UNLOGGED


def test_high_priority_with_no_budget_guard_is_high_priority_inert(
    tmp_path: Path,
) -> None:
    """Only a logged inert ``no_budget`` line keeps ``high_priority_inert``."""
    moment = datetime(2026, 11, 1, 22, 0, 0)
    session = "sess-high-no-budget"
    _write_log(
        tmp_path, "proxy.log",
        [
            _priority(moment, session, "high"),
            _inert_no_budget(moment, session),
            _eviction(moment + timedelta(seconds=121), session, new_tokens=20723,
                      eta_prefill_s="278.8", elapsed_s=121.0),
        ],
    )

    summary = _attribute(tmp_path)
    assert summary["by_admission"] == {attribution.HIGH_PRIORITY_INERT: 1}
    assert summary["rows"][0]["admission"] == attribution.HIGH_PRIORITY_INERT


def test_summary_reports_budget_and_over_deadline_evidence(log_dir: Path) -> None:
    summary = _attribute(log_dir)

    assert summary["all_within_cold_prefill_budget"] is True
    # Only the ``high`` priority event carried a priced ETA above the deadline:
    # the pricing was right and the bound was never enforced.
    assert summary["priced_eta_over_deadline"] == 1
    # The priced new tokens per path are the report's wasted-token figure.
    assert sum(summary["new_tokens_by_admission"].values()) == 12000 + 73 + 14041 + 20723 + 14041


def test_unknown_and_missing_fields_never_crash(tmp_path: Path) -> None:
    moment = datetime(2026, 11, 1, 12, 0, 0)
    _write_log(
        tmp_path, "proxy.log",
        [
            _eviction(moment, "sess-unknown", eta_prefill_s="unknown",
                      elapsed_s=0.0),
        ],
    )

    summary = _attribute(tmp_path)
    assert summary["evictions"] == 1
    row = summary["rows"][0]
    assert row["priced_eta_s"] is None
    assert row["admission"] == attribution.INERT_UNLOGGED
    assert summary["by_priority"] == {"unknown": 1}


def test_cli_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    moment = datetime(2026, 11, 1, 22, 0, 0)
    _write_log(
        tmp_path, "proxy.log",
        [
            _priority(moment, "sess-cli", "low"),
            _guard(moment, "sess-cli", priority="low"),
            _eviction(moment + timedelta(seconds=121), "sess-cli"),
        ],
    )

    exit_code = attribution.main([
        "--log-dir", str(tmp_path),
        "--start", "2026-11-01 11:00:00",
        "--until", "2026-11-02 11:00:00",
        "--json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_admission"] == {attribution.RUNG_ACCEPTED: 1}


def test_cli_rejects_an_inverted_window(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = attribution.main([
        "--start", "2026-11-02 11:00:00",
        "--until", "2026-11-01 11:00:00",
    ])

    assert exit_code == 2
    assert "--start must be before --until" in capsys.readouterr().out
