"""Wasted-GPU-compute surface in the usage report (LP-0MW2SQNHJ0044PV6).

The report already counts post-acceptance evictions; it could not answer *how
much GPU compute the window burned on work nobody received, and why*. This
module verifies the new surface end to end:

- AC1 the analysis **reuses** the committed analyser
  (``proxy/scripts/analyse_prefill_deadline_residual.py``) — a real synthetic
  window is driven through it and the normalised result carries its figures;
- AC2 the summary table renders **before** the first data section;
- AC3 the detail section renders the per-reason, attribution, waste and
  release→cancel tables;
- AC4 the appendix renders after every other section and states the basis,
  thresholds and caveats;
- AC5 ``--json`` carries the same figures;
- AC6 degradation: no analyser / no logs / no llama-server log / a coverage
  mismatch each render a caveat instead of a wrong number;
- the restart-free determination — the check whose absence produced a wrong
  "clean window" claim on 2026-11-04 — flags a window containing a backend
  restart.

Every test drives behaviour through the module's public API or the rendered
report; none reads the implementation source.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import reporting  # noqa: E402
import wasted_compute  # noqa: E402

WINDOW_START = datetime(2026, 11, 4, 16, 0, 0)
WINDOW_END = datetime(2026, 11, 5, 2, 0, 0)  # 10 h
EXPECTED_HOURS = 10.0


# ---------------------------------------------------------------------------
# Fixtures: synthetic proxy.log lines in the real structured format
# ---------------------------------------------------------------------------


def _eviction(
    ts: str,
    session: str,
    reason: str,
    *,
    estimated_tokens: int = 60_000,
    new_tokens: int = 3_000,
    elapsed_s: float = 180.1,
    eta_prefill_s: str = "95.0",
) -> str:
    return (
        f"{ts} - ERROR - local_eviction_after_acceptance session={session} "
        f"reason={reason} estimated_tokens={estimated_tokens} "
        f"new_tokens={new_tokens} eta_prefill_s={eta_prefill_s} "
        f"elapsed_s={elapsed_s} slot_id=unknown llama_task_id=unknown "
        "endpoint=default"
    )


#: Three deadline events (two user, one audit), one user cache-deficit abort
#: and one user backend-restart eviction. All deadline events price 5 % of the
#: prompt as new compute and show no backend progress, so they attribute to
#: predicted-warmth.
WINDOW_LINES = [
    _eviction("2026-11-04 17:10:00,000", "herdr-1", "prefill_deadline"),
    _eviction("2026-11-04 18:20:00,000", "herdr-2", "prefill_deadline"),
    _eviction("2026-11-04 19:30:00,000", "audit-1", "prefill_deadline"),
    _eviction(
        "2026-11-04 20:40:00,000",
        "herdr-3",
        "measured_cache_deficit",
        estimated_tokens=40_000,
        new_tokens=400,
    ),
    _eviction(
        "2026-11-04 21:50:00,000",
        "herdr-4",
        "backend_restart",
        estimated_tokens=32_347,
        new_tokens=16_816,
        elapsed_s=3.1,
    ),
]

LLAMA_LINE = (
    "srv  log_server_r: done request: POST /v1/chat/completions 127.0.0.1 200"
)


def _write_window(tmp_path: Path, *, llama_log: bool = False) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text("\n".join(WINDOW_LINES) + "\n", encoding="utf-8")
    if llama_log:
        (log_dir / "llama-server.log").write_text(LLAMA_LINE + "\n", encoding="utf-8")
    return log_dir


def _analyse(log_dir: Path, **kwargs) -> wasted_compute.WastedCompute:
    return wasted_compute.analyse_window(
        log_dir, WINDOW_START, WINDOW_END, llama_log_dir=log_dir, **kwargs
    )


def _empty_summary():
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate([], WINDOW_START, WINDOW_END, schedule)


# ---------------------------------------------------------------------------
# AC1 — analysis reuses the committed analyser
# ---------------------------------------------------------------------------


def test_analysis_reuses_the_committed_analyser(tmp_path):
    result = _analyse(_write_window(tmp_path))

    assert result.status == "ok", result.reason
    assert result.hours == pytest.approx(EXPECTED_HOURS, abs=1e-6)
    # The figures come from the analyser's own reason taxonomy and session
    # classifier: 3 deadline events (2 user, 1 audit) and 1 user cache abort.
    deadline = result.reason_row("prefill_deadline")
    assert deadline is not None
    assert (deadline.total, deadline.user, deadline.audit) == (3, 2, 1)
    assert deadline.user_per_hour == pytest.approx(0.2, abs=1e-6)
    deficit = result.reason_row("measured_cache_deficit")
    assert deficit is not None
    assert (deficit.total, deficit.user) == (1, 1)
    assert result.total_events == 5
    assert result.events_per_hour == pytest.approx(0.5, abs=1e-6)


def test_composite_rate_counts_both_reasons_and_excludes_audit(tmp_path):
    result = _analyse(_write_window(tmp_path))

    # The conversion-proof composite: user deadline (2) + user cache deficit
    # (1); the audit deadline is excluded, the backend restart is not a
    # prefill failure and is excluded too.
    assert result.composite_user_events == 3
    assert result.composite_user_per_hour == pytest.approx(0.3, abs=1e-6)
    assert result.rate_target_per_hour == pytest.approx(0.5)
    assert result.rate_ok is True


# ---------------------------------------------------------------------------
# Attribution, guards and the restart-free determination
# ---------------------------------------------------------------------------


def test_attribution_and_guards_are_normalised(tmp_path):
    result = _analyse(_write_window(tmp_path))

    # All three deadline events priced 5 % of the prompt as new compute and
    # reported no backend progress: predicted-warmth, no tenant present.
    assert result.attribution_total == 3
    assert result.attribution["predicted-warmth"] == 3
    assert result.attribution["occupancy"] == 0
    assert result.attribution["capacity"] == 0
    assert result.occupancy_share == pytest.approx(0.0)
    assert result.capacity_share == pytest.approx(0.0)
    assert result.occupancy_ok is True
    assert result.capacity_ok is True


def test_backend_restart_in_window_is_not_restart_free(tmp_path):
    result = _analyse(_write_window(tmp_path))

    assert result.proxy_restarts == 0
    assert result.backend_restart_evictions == 1
    assert result.restart_free is False
    # The rate passes and both guards pass, but a restart re-prices capacity
    # and discards prefills, so the window cannot certify anything.
    assert result.verdict == "inconclusive"
    assert any("backend restart" in caveat for caveat in result.caveats)


def test_guard_breaches_fail_the_window():
    dataset = {
        "window": {"hours": 24.0, "start": "2026-11-01 00:00:00", "end": "2026-11-02 00:00:00"},
        "source": {"restarts_in_window": []},
        "evictions": {"total": 12, "by_reason": {}, "per_hour": 0.5},
        "reason_split": {
            "prefill_deadline": {
                "total": 12,
                "user": 12,
                "audit": 0,
                "total_per_hour": 0.5,
                "user_per_hour": 0.5,
            }
        },
        "attribution": {
            "capture_available": True,
            "totals": {"occupancy": 0, "capacity": 8, "predicted-warmth": 4},
            "abandoned_prefill_overlap": {"events": 0, "total": 12},
        },
        "events": [{"new_tokens": 1_000} for _ in range(12)],
        "abandoned_prefill_windows": {"coverage": {"verdict": "exact", "entries": 1, "releases": 1}},
    }

    result = wasted_compute.normalise(dataset)

    assert result.rate_ok is True  # 0.5/h is exactly at the target
    assert result.capacity_share == pytest.approx(8 / 12)
    assert result.capacity_ok is False  # ≥ 50 % trips the floor
    assert result.verdict == "fail"


def test_occupancy_share_above_the_guard_fails_the_window():
    dataset = {
        "window": {"hours": 24.0, "start": "2026-11-01 00:00:00", "end": "2026-11-02 00:00:00"},
        "source": {"restarts_in_window": []},
        "evictions": {"total": 4, "by_reason": {}, "per_hour": 0.17},
        "reason_split": {
            "prefill_deadline": {
                "total": 4,
                "user": 4,
                "audit": 0,
                "total_per_hour": 0.17,
                "user_per_hour": 0.17,
            }
        },
        "attribution": {
            "capture_available": True,
            "totals": {"occupancy": 2, "capacity": 0, "predicted-warmth": 2},
            "abandoned_prefill_overlap": {"events": 0, "total": 4},
        },
        "events": [],
        "abandoned_prefill_windows": {"coverage": {"verdict": "exact", "entries": 4, "releases": 4}},
    }

    result = wasted_compute.normalise(dataset)

    assert result.occupancy_share == pytest.approx(0.5)
    assert result.occupancy_ok is False
    assert result.verdict == "fail"


def test_clean_window_below_every_threshold_passes():
    dataset = {
        "window": {"hours": 24.0, "start": "2026-11-01 00:00:00", "end": "2026-11-02 00:00:00"},
        "source": {"restarts_in_window": []},
        "evictions": {"total": 2, "by_reason": {}, "per_hour": 0.08},
        "reason_split": {
            "prefill_deadline": {
                "total": 2,
                "user": 2,
                "audit": 0,
                "total_per_hour": 0.08,
                "user_per_hour": 0.08,
            }
        },
        "attribution": {
            "capture_available": True,
            "totals": {"occupancy": 0, "capacity": 0, "predicted-warmth": 2},
            "abandoned_prefill_overlap": {"events": 0, "total": 2},
        },
        "events": [],
        "abandoned_prefill_windows": {
            "coverage": {"verdict": "exact", "entries": 2, "releases": 2}
        },
    }

    result = wasted_compute.normalise(dataset)

    assert result.restart_free is True
    assert result.verdict == "pass"


# ---------------------------------------------------------------------------
# Waste accounting and degradation
# ---------------------------------------------------------------------------


def test_waste_sums_priced_tokens_and_post_cancel_evaluation(tmp_path):
    result = _analyse(_write_window(tmp_path))

    # Priced new-token waste at the deadline is exact from the proxy log
    # alone: 3 000 x 3 deadline events (the cache-deficit abort is a *priced*
    # detour too, but the analyser reports per-event tokens for deadline
    # events only, and an early abort's loss is bounded by design). Tokens
    # evaluated after the cancel need the backend log, so this window
    # contributes 0 and says so.
    assert result.priced_tokens_discarded == 9_000
    assert result.tokens_after_cancel == 0
    assert result.wasted_tokens == 9_000
    assert result.coverage_verdict is None
    assert any("no llama-server log" in caveat for caveat in result.caveats)


def test_backend_figures_are_caveated_when_coverage_is_not_exact(tmp_path):
    result = _analyse(_write_window(tmp_path, llama_log=True))

    # A llama-server log whose segment count cannot be matched to the release
    # count is 'mismatch': the per-release figures stay indicative.
    assert result.coverage_verdict is not None
    assert result.coverage_exact is False
    assert any("indicative" in caveat for caveat in result.caveats)


def test_missing_logs_degrade_without_raising(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = wasted_compute.analyse_window(empty, WINDOW_START, WINDOW_END)

    assert result.status == "unavailable"
    assert result.available is False
    assert "proxy.log" in result.reason


def test_missing_analyser_degrades_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv(wasted_compute.ANALYSER_ENV, str(tmp_path / "does-not-exist.py"))

    result = wasted_compute.analyse_window(
        _write_window(tmp_path), WINDOW_START, WINDOW_END
    )

    assert result.status == "unavailable"
    assert "analyser not found" in result.reason


# ---------------------------------------------------------------------------
# AC2/AC3/AC4 — rendering order and content
# ---------------------------------------------------------------------------


def test_report_renders_summary_detail_and_appendix_in_order(tmp_path):
    result = _analyse(_write_window(tmp_path))
    markdown = reporting.build_report(_empty_summary(), None, wasted=result)

    summary = markdown.index("## Wasted GPU compute (summary)\n")
    detail = markdown.index("## Wasted GPU compute\n")
    appendix = markdown.index("## Appendix: wasted GPU compute")

    # Summary near the start: before the first data section.
    assert summary < markdown.index("## Summary by hour\n")
    assert summary < detail < appendix
    # The appendix is last: after the report's own notes section.
    assert appendix > markdown.index("## Notes and limitations\n")

    # Summary table content: the composite rate against its target and the
    # restart-free check that failed for this window.
    assert "| User-class prefill failures" in markdown
    assert "(0.30/h)" in markdown
    assert "| Restart-free window | no — 0 proxy / 1 backend restart(s) | required | FAIL |" in markdown
    assert "**NOT CERTIFIED**" in markdown

    # Detail section: one row per reason, attribution and waste tables.
    assert "| `prefill_deadline` | 3 | 2 | 1 | 0.30 | 0.20 |" in markdown
    assert "| predicted-warmth | 3 | 100% |" in markdown
    assert "| Total wasted GPU tokens | 9000 |" in markdown
    assert "| Cancel → backend stop (p50 / p90) |" in markdown

    # Appendix: basis, thresholds and caveats are stated, not implied.
    assert "**Measurement basis.**" in markdown
    assert "≤ 0.5 user-class events/h" in markdown
    assert "**Caveats.**" in markdown


def test_report_omits_the_section_when_no_analysis_was_run():
    markdown = reporting.build_report(_empty_summary(), None)

    assert "Wasted GPU compute" not in markdown


def test_unavailable_analysis_renders_a_named_reason_not_a_table(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = wasted_compute.analyse_window(empty, WINDOW_START, WINDOW_END)

    markdown = reporting.build_report(_empty_summary(), None, wasted=result)

    assert "## Wasted GPU compute (summary)\n" in markdown
    assert "_Not available for this window:" in markdown
    assert "| Figure | Value | Target | Verdict |" not in markdown


# ---------------------------------------------------------------------------
# AC5 — machine-readable block
# ---------------------------------------------------------------------------


def test_json_block_carries_the_same_figures(tmp_path):
    result = _analyse(_write_window(tmp_path))

    data = reporting.summary_to_json(_empty_summary(), None, None, wasted=result)
    block = data["wasted_compute"]

    assert block["status"] == "ok"
    assert block["composite_user_events"] == 3
    assert block["composite_user_per_hour"] == pytest.approx(0.3, abs=1e-6)
    assert block["rate_ok"] is True
    assert block["restart_free"] is False
    assert block["verdict"] == "inconclusive"
    assert block["waste"] == {
        "priced_tokens_discarded": 9_000,
        "tokens_evaluated_after_cancel": 0,
        "wasted_tokens": 9_000,
        "backend_completed_prefills": None,
        "client_consumed_local_streams": None,
    }
    # Additive only: the pre-existing eviction keys keep their meaning (this
    # synthetic summary has no events, so the counter is empty but present).
    assert data["local_evictions_after_acceptance_by_reason"] == {}


def test_json_block_is_null_without_the_analysis():
    data = reporting.summary_to_json(_empty_summary(), None, None)

    assert data["wasted_compute"] is None
