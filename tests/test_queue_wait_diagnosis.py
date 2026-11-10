#!/usr/bin/env python3
"""Tests for the per-wait queue-abandonment diagnosis (``LP-0MVV29XPT003OXI5``).

The diagnosis is what the parent item's open questions are answered from, so
its correlation and classification rules are pinned here from **synthetic log
lines only** — deterministic, no live log dependency, no sleeps.

Covered:

- AC1: one row per queue wait carrying enqueue time, trigger, observed wait,
  cause, the same-request guard projection fields and the blocker class; an
  ambiguous correlation is flagged ``ambiguous=true`` and never silently joined.
- AC2(a): a wait whose same-request admit exists is joined (``own_request``).
- AC2(b): a wait whose only admission belongs to a *different* request in the
  same session is **not** joined and is flagged ``ambiguous=true``.
- AC2(c): a wait with no admission line at all is reported with no guard and is
  **not** flagged ambiguous (there was nothing ambiguous to resolve).
- AC2(d): blocker classification ``prefill_in_flight`` vs
  ``lease_holder_generating`` vs ``unknown``, including the lookback bound, a
  released lease clearing the holder, and the self-blocked flag.
- AC3(c): the pre-fix banding rule is reproduced per wait (``legacy_band``), so
  the phantom ``32k-64k`` row is proven to be an attribution artefact rather
  than a genuine band.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from queue_wait_diagnosis import (
    BLOCKER_GENERATING,
    BLOCKER_PREFILL,
    BLOCKER_UNKNOWN,
    CORRELATION_MISMATCH,
    CORRELATION_NONE,
    CORRELATION_OWN,
    CORRELATION_SESSION,
    analyse,
    blocker_at,
    build_rows,
    collect,
    main,
    pair_waits,
    parse_lease_event,
    projected_bucket,
)

SINCE = datetime(2026, 10, 19, 0, 0, 0)
UNTIL = datetime(2026, 10, 20, 0, 0, 0)


# ---------------------------------------------------------------------------
# Synthetic log-line builders (the deployed message shapes)
# ---------------------------------------------------------------------------


def _admission(
    ts: str,
    *,
    session: str = "s-1",
    new_tokens: int = 6000,
    decision: str = "admit",
    projected: float = 40.0,
    eta_prefill: float = 20.0,
    eta_occupancy: float = 20.0,
    busy_slots: int = 1,
    in_flight: int = 1,
    budget: float = 60.0,
    rate_bucket: str = "contended",
) -> str:
    return (
        f"{ts} - llama-proxy - INFO - local_admission_guard "
        f"decision={decision} reason=within_budget enforced=false "
        f"eta_prefill_s={eta_prefill:.1f} eta_occupancy_s={eta_occupancy:.1f} "
        f"projected_wait_s={projected:.1f} budget_s={budget:.1f} "
        f"new_tokens={new_tokens} budget_tokens=25000 "
        f"rate_bucket={rate_bucket} contended_samples=12 "
        f"busy_slots={busy_slots} in_flight_prefills={in_flight} "
        f"free_slot=false priority=low policy=reject model=Qwen3 samples=12 "
        f"provider=local-qwen3 session={session}"
    )


def _enqueue(
    ts: str,
    *,
    session: str | None = None,
    new_tokens: int | None = None,
    trigger: str | None = None,
) -> str:
    fields = "max_wait=60.0s max_depth=8 saturation=prefill"
    if session is not None:
        fields += f" session={session}"
    if new_tokens is not None:
        fields += f" new_tokens={new_tokens} rate_bucket=contended"
    if trigger is not None:
        fields += f" trigger={trigger}"
    return f"{ts} - llama-proxy - INFO - contention_queue_wait {fields}"


def _fallback(
    ts: str,
    *,
    session: str = "s-1",
    queued: float = 60.0,
    reason: str | None = None,
    trigger: str | None = None,
    cause: str | None = None,
    new_tokens: int | None = None,
) -> str:
    fields = (
        f"provider=local-qwen3 session={session} queued_duration={queued:.2f}s "
        "preference=local-preferred"
    )
    if reason is not None:
        fields += f" reason={reason}"
    if trigger is not None:
        fields += f" trigger={trigger}"
    if cause is not None:
        fields += f" cause={cause}"
    if new_tokens is not None:
        fields += f" new_tokens={new_tokens}"
    return f"{ts} - llama-proxy - INFO - contention_queue_fallback_after_queue {fields}"


def _dispatch(
    ts: str,
    *,
    session: str = "s-1",
    queued: float = 12.0,
    trigger: str | None = None,
    new_tokens: int | None = None,
) -> str:
    fields = (
        f"provider=local-qwen3 session={session} queued_duration={queued:.2f}s "
        "policy=queue depth=1"
    )
    if trigger is not None:
        fields += f" trigger={trigger}"
    if new_tokens is not None:
        fields += f" new_tokens={new_tokens}"
    return f"{ts} - llama-proxy - INFO - contention_queue_dispatch {fields}"


def _first_byte(ts: str, *, session: str = "holder") -> str:
    return (
        f"{ts} - llama-proxy - INFO - dispatch_first_byte_ms=512.0 "
        f"dispatch_to_first_byte_ms=511.4 cache_state=warm cached_ratio=0.990 "
        f"session={session} model=Qwen3 priority=medium"
    )


def _lease_prefill(ts: str, *, session: str = "holder") -> str:
    return (
        f"{ts} - llama-proxy - INFO - lease_extended_during_prefill "
        f"session={session} progress=None alive=False in_flight=True buffer=30s"
    )


def _lease_release(ts: str, *, session: str = "holder", reason: str = "request_completed") -> str:
    return f"{ts} - llama-proxy - INFO - lease_released session={session} reason={reason}"


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "proxy.log").write_text("\n".join(lines) + "\n")
    return log_dir


def _report(tmp_path: Path, lines: list[str], **kwargs) -> dict:
    return analyse(_write_log(tmp_path, lines), SINCE, UNTIL, **kwargs)


def _rows(tmp_path: Path, lines: list[str], **kwargs) -> list[dict]:
    return _report(tmp_path, lines, **kwargs)["rows"]


# ---------------------------------------------------------------------------
# AC1 / AC2(a): one row per wait, same-request guard projection
# ---------------------------------------------------------------------------


def test_row_carries_wait_guard_and_blocker_fields(tmp_path):
    """AC1: one row per queue wait with every diagnosed field populated."""
    lines = [
        _lease_prefill("2026-10-19 09:50:00,000", session="holder"),
        _admission(
            "2026-10-19 09:51:00,000",
            session="s-1",
            new_tokens=9000,
            projected=35.8,
            eta_prefill=17.9,
            eta_occupancy=17.9,
            busy_slots=1,
            in_flight=1,
            rate_bucket="contended",
        ),
        _enqueue(
            "2026-10-19 09:51:05,000",
            session="s-1",
            new_tokens=9000,
            trigger="concurrency_limit",
        ),
        _fallback(
            "2026-10-19 09:52:05,000",
            session="s-1",
            queued=60.0,
            trigger="concurrency_limit",
            cause="queue_wait_expired",
            new_tokens=9000,
        ),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "s-1"
    assert row["enqueue_ts"] == "2026-10-19 09:51:05"
    assert row["outcome"] == "abandoned"
    assert row["trigger"] == "concurrency_limit"
    assert row["cause"] == "queue_wait_expired"
    assert row["observed_wait_s"] == 60.0
    assert row["own_new_tokens"] == 9000
    # AC1/AC2(a): same-request admit joined, positive own-request evidence.
    assert row["correlation"] == CORRELATION_OWN
    assert row["ambiguous"] is False
    assert row["guard"] == {
        "decision": "admit",
        "reason": "within_budget",
        "projected_wait_s": 35.8,
        "eta_prefill_s": 17.9,
        "eta_occupancy_s": 17.9,
        "busy_slots": 1,
        "in_flight_prefills": 1,
        "rate_bucket": "contended",
        "new_tokens": 9000,
        "budget_s": 60.0,
    }
    # AC1/AC2(d): the lease holder was still prefilling at the enqueue.
    assert row["blocker_class"] == BLOCKER_PREFILL
    assert row["blocker_session"] == "holder"
    assert row["blocker_is_self"] is False
    assert row["band"] == "8k-20k"
    assert row["band_evidence"] == "own_outcome"


def test_own_request_needs_matching_new_tokens_not_merely_precedence(tmp_path):
    """AC2(a): a preceding admit for the same session is joined when it matches."""
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=2500),
        _admission("2026-10-19 09:05:00,000", session="s-1", new_tokens=12000),
        _enqueue("2026-10-19 09:05:10,000", session="s-1", new_tokens=12000),
        _dispatch("2026-10-19 09:05:40,000", session="s-1", queued=30.0, new_tokens=12000),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    assert rows[0]["correlation"] == CORRELATION_OWN
    assert rows[0]["guard"]["new_tokens"] == 12000
    assert rows[0]["ambiguous"] is False


# ---------------------------------------------------------------------------
# AC2(b) / AC2(c): never join another request's admission
# ---------------------------------------------------------------------------


def test_admission_for_a_different_request_is_not_joined(tmp_path):
    """AC2(b): another request's admission in the same session is not borrowed."""
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=45000),
        _enqueue("2026-10-19 09:01:00,000", session="s-1", new_tokens=1500),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0, new_tokens=1500),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    row = rows[0]
    assert row["correlation"] == CORRELATION_MISMATCH
    assert row["ambiguous"] is True
    assert row["guard"] is None, "a different request's projection must never be joined"
    assert row["candidates"] == 1
    # Banding also refuses the borrowed evidence (the phantom-band defect).
    assert row["band"] == "<8k"
    assert row["band_evidence"] == "own_outcome"


def test_wait_with_no_admission_is_not_ambiguous(tmp_path):
    """AC2(c): nothing to correlate is reported as ``none``, not ambiguous."""
    lines = [
        _enqueue("2026-10-19 09:01:00,000", session="s-1", new_tokens=1500),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0, new_tokens=1500),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    assert rows[0]["correlation"] == CORRELATION_NONE
    assert rows[0]["ambiguous"] is False
    assert rows[0]["guard"] is None
    assert rows[0]["candidates"] == 0


def test_session_fallback_join_is_always_flagged_ambiguous(tmp_path):
    """AC1: the legacy fallback join is reported, never silently applied."""
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=6000),
        # Pre-fix log shape: neither queue line carries its own pricing.
        _enqueue("2026-10-19 09:01:00,000"),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    row = rows[0]
    assert row["correlation"] == CORRELATION_SESSION
    assert row["ambiguous"] is True
    assert row["guard"]["projected_wait_s"] == 40.0
    assert row["candidates"] == 1
    assert row["band_evidence"] == "session_chain"
    assert row["band"] == "<8k"


def test_an_admission_is_consumed_by_only_one_wait(tmp_path):
    """AC1: the same admission cannot be the projection for two waits."""
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=6000),
        _enqueue("2026-10-19 09:01:00,000", session="s-1"),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0),
        _enqueue("2026-10-19 09:03:00,000", session="s-1"),
        _fallback("2026-10-19 09:04:00,000", session="s-1", queued=60.0),
    ]
    rows = _rows(tmp_path, lines)

    assert [row["correlation"] for row in rows] == [
        CORRELATION_SESSION,
        CORRELATION_NONE,
    ]
    assert rows[1]["guard"] is None


# ---------------------------------------------------------------------------
# AC2(d): blocker classification
# ---------------------------------------------------------------------------


def test_blocker_class_distinguishes_prefill_generating_and_unknown(tmp_path):
    """AC2(d): the three blocker classes from lease/dispatch evidence."""
    lines = [
        _lease_prefill("2026-10-19 09:50:00,000", session="holder-a"),
        _enqueue("2026-10-19 09:51:00,000", session="w-1"),
        _dispatch("2026-10-19 09:51:05,000", session="w-1", queued=5.0),
        _first_byte("2026-10-19 09:52:00,000", session="holder-a"),
        _enqueue("2026-10-19 09:53:00,000", session="w-2"),
        _dispatch("2026-10-19 09:53:05,000", session="w-2", queued=5.0),
        _lease_release("2026-10-19 09:54:00,000", session="holder-a"),
        _enqueue("2026-10-19 09:55:00,000", session="w-3"),
        _dispatch("2026-10-19 09:55:05,000", session="w-3", queued=5.0),
    ]
    rows = _rows(tmp_path, lines)

    assert [row["session"] for row in rows] == ["w-1", "w-2", "w-3"]
    assert [row["blocker_class"] for row in rows] == [
        BLOCKER_PREFILL,
        BLOCKER_GENERATING,
        BLOCKER_UNKNOWN,
    ]
    assert [row["blocker_session"] for row in rows] == ["holder-a", "holder-a", None]


def test_blocker_lookback_bounds_stale_lease_evidence(tmp_path):
    """AC2(d): evidence older than the lookback is not treated as the holder."""
    lines = [
        _lease_prefill("2026-10-19 08:00:00,000", session="holder-a"),
        _enqueue("2026-10-19 09:00:00,000", session="w-1"),
        _dispatch("2026-10-19 09:00:05,000", session="w-1", queued=5.0),
    ]
    stale = _rows(tmp_path, lines)
    assert stale[0]["blocker_class"] == BLOCKER_UNKNOWN

    wide = _rows(tmp_path, lines, blocker_lookback_s=7200.0)
    assert wide[0]["blocker_class"] == BLOCKER_PREFILL


def test_self_blocked_wait_is_flagged(tmp_path):
    """AC2(d): the only candidate holder is the waiting session itself."""
    lines = [
        _lease_prefill("2026-10-19 09:50:00,000", session="s-1"),
        _enqueue("2026-10-19 09:51:00,000", session="s-1"),
        _fallback(
            "2026-10-19 09:52:00,000",
            session="s-1",
            queued=60.0,
            reason="local_lease_active",
        ),
    ]
    rows = _rows(tmp_path, lines)

    assert rows[0]["blocker_session"] == "s-1"
    assert rows[0]["blocker_is_self"] is True
    assert rows[0]["blocker_class"] == BLOCKER_PREFILL
    # The legacy ``reason=local_lease_active`` is unambiguous trigger evidence.
    assert rows[0]["trigger"] == "local_lease_active"


def test_lease_marker_parsing_is_anchored_to_the_message_body(tmp_path):
    """A request body that merely quotes a marker is never lease evidence."""
    body = (
        '2026-10-19 09:50:00,000 - INFO - [remote] POST body={"text": '
        '"- INFO - lease_extended_during_prefill session=holder"}'
    )
    assert parse_lease_event(body, datetime(2026, 10, 19, 9, 50, 0)) is None

    real = _lease_prefill("2026-10-19 09:50:00,000", session="holder")
    event = parse_lease_event(real, datetime(2026, 10, 19, 9, 50, 0))
    assert event is not None
    assert event.kind == "prefill"
    assert event.session == "holder"


def test_blocker_at_returns_newest_live_holder(tmp_path):
    """AC2(d): the most recent lease evidence wins, per session."""
    leases = collect(
        _write_log(
            tmp_path,
            [
                _lease_prefill("2026-10-19 09:40:00,000", session="holder-a"),
                _first_byte("2026-10-19 09:50:00,000", session="holder-b"),
            ],
        ),
        SINCE,
        UNTIL,
    ).leases

    assert blocker_at(datetime(2026, 10, 19, 9, 51), leases) == (
        BLOCKER_GENERATING,
        "holder-b",
    )


# ---------------------------------------------------------------------------
# Pairing: one row per wait even when the enqueue line carries no session
# ---------------------------------------------------------------------------


def test_enqueue_is_paired_by_time_when_it_carries_no_session(tmp_path):
    """The 2026-10-29/30 shape logs no session on the enqueue line."""
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=6000),
        _enqueue("2026-10-19 09:01:00,000"),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0),
    ]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    assert rows[0]["enqueue_ts"] == "2026-10-19 09:01:00"
    assert rows[0]["outcome_ts"] == "2026-10-19 09:02:00"
    assert rows[0]["session"] == "s-1"


def test_enqueue_without_outcome_is_still_one_row(tmp_path):
    """AC1: a wait with no outcome in the window is emitted, not dropped."""
    lines = [_enqueue("2026-10-19 09:01:00,000", session="s-1", new_tokens=1500)]
    rows = _rows(tmp_path, lines)

    assert len(rows) == 1
    assert rows[0]["outcome"] is None
    assert rows[0]["observed_wait_s"] is None
    assert rows[0]["cause"] is None
    assert rows[0]["enqueue_ts"] == "2026-10-19 09:01:00"


def test_enqueue_pairing_ignores_a_distant_enqueue(tmp_path):
    """Pairing is time-bounded: a far-away enqueue is not borrowed."""
    lines = [
        _enqueue("2026-10-19 06:00:00,000", session="s-1"),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0),
    ]
    log_dir = _write_log(tmp_path, lines)
    collected = collect(log_dir, SINCE, UNTIL)
    waits = pair_waits(collected.enqueues, collected.outcomes)

    assert len(waits) == 2
    paired = [wait for wait in waits if wait.outcome is not None]
    assert paired[0].enqueue is None
    # The enqueue instant is still recovered from the outcome's own duration.
    assert paired[0].enqueue_ts == datetime(2026, 10, 19, 9, 1)


# ---------------------------------------------------------------------------
# AC3(c): the phantom band is an attribution artefact, per wait
# ---------------------------------------------------------------------------


def test_legacy_band_reproduces_the_phantom_and_the_fix_moves_the_wait(tmp_path):
    """AC3(c): a wait with only a *later* admission is not banded by it."""
    lines = [
        _enqueue("2026-10-19 09:01:00,000", session="s-1", new_tokens=1500),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0, new_tokens=1500),
        # The session's only admission arrives 4 m 38 s *after* the abandonment.
        _admission("2026-10-19 09:06:38,000", session="s-1", new_tokens=45000, decision="reject"),
    ]
    report = _report(tmp_path, lines)
    row = report["rows"][0]

    assert row["legacy_band"] == "32k-64k", "the pre-fix rule borrowed the later admission"
    assert row["band"] == "<8k"
    assert report["summary"]["rebanded"] == 1
    assert report["summary"]["legacy_bands"]["32k-64k"]["abandoned"] == 1
    assert "32k-64k" not in report["summary"]["bands"]


def test_legacy_band_falls_back_to_the_latest_admission_when_none_precedes(tmp_path):
    """AC3(c): with no own pricing the pre-fix rule bands by the later admit."""
    lines = [
        _enqueue("2026-10-19 09:01:00,000"),
        _fallback("2026-10-19 09:02:00,000", session="s-1", queued=60.0),
        _admission("2026-10-19 09:06:38,000", session="s-1", new_tokens=45000, decision="reject"),
    ]
    row = _rows(tmp_path, lines)[0]

    assert row["legacy_band"] == "32k-64k"
    assert row["band"] == "unknown"
    assert row["band_evidence"] == "unknown"


# ---------------------------------------------------------------------------
# Summary arithmetic
# ---------------------------------------------------------------------------


def test_summary_splits_by_outcome_trigger_and_blocker(tmp_path):
    """AC1/AC3: the summary is the record's arithmetic, from the rows."""
    lines = [
        _lease_prefill("2026-10-19 08:59:00,000", session="holder-a"),
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=6000, projected=28.0),
        _enqueue("2026-10-19 09:00:05,000", session="s-1", trigger="concurrency_limit"),
        _fallback(
            "2026-10-19 09:01:05,000",
            session="s-1",
            queued=60.0,
            trigger="concurrency_limit",
            cause="queue_wait_expired",
        ),
        _admission("2026-10-19 09:10:00,000", session="s-2", new_tokens=2000, projected=12.0),
        _enqueue("2026-10-19 09:10:05,000", session="s-2", trigger="local_lease_active"),
        _dispatch("2026-10-19 09:10:25,000", session="s-2", queued=20.0, trigger="local_lease_active"),
    ]
    summary = _report(tmp_path, lines)["summary"]

    assert summary["queue_waits"] == 2
    assert summary["by_outcome"] == {"abandoned": 1, "dispatched": 1}
    assert summary["abandoned_by_trigger"] == {"concurrency_limit": 1}
    assert summary["abandoned_by_cause"] == {"queue_wait_expired": 1}

    abandoned = summary["abandoned_guard"]
    assert abandoned["with_guard"] == 1
    assert abandoned["admits"] == 1
    assert abandoned["admits_below_budget"] == 1
    assert abandoned["median_projected_s"] == 28.0
    assert abandoned["median_observed_s"] == 60.0
    assert abandoned["observed_minus_projected_s"]["median"] == 32.0
    assert abandoned["projected_buckets"] == {"20-30": 1}
    assert abandoned["occupancy_is_symmetric_floor"] == 1
    assert abandoned["occupancy_raised_by_measurement"] == 0

    dispatched = summary["dispatched_guard"]
    assert dispatched["median_projected_s"] == 12.0
    assert dispatched["median_observed_s"] == 20.0

    assert summary["blocker"]["abandoned_by_class"] == {
        BLOCKER_PREFILL: 1,
        BLOCKER_GENERATING: 0,
        BLOCKER_UNKNOWN: 0,
    }
    assert summary["blocker"]["abandoned_by_class_and_trigger"] == {
        "concurrency_limit": {BLOCKER_PREFILL: 1}
    }
    assert summary["bands"]["<8k"] == {
        "queued": 2,
        "dispatched": 1,
        "abandoned": 1,
        "no_outcome": 0,
    }


def test_summary_detects_an_occupancy_term_raised_by_measurement(tmp_path):
    """The floor/measurement discriminator separates the two occupancy shapes."""
    lines = [
        _admission(
            "2026-10-19 09:00:00,000",
            session="s-1",
            eta_prefill=12.0,
            eta_occupancy=48.0,
            projected=60.0,
        ),
        _enqueue("2026-10-19 09:00:05,000", session="s-1"),
        _fallback("2026-10-19 09:01:05,000", session="s-1", queued=60.0),
    ]
    abandoned = _report(tmp_path, lines)["summary"]["abandoned_guard"]

    assert abandoned["occupancy_raised_by_measurement"] == 1
    assert abandoned["occupancy_is_symmetric_floor"] == 0


def test_projected_bucket_boundaries():
    assert projected_bucket(None) == "unknown"
    assert projected_bucket(0.0) == "<20"
    assert projected_bucket(19.99) == "<20"
    assert projected_bucket(20.0) == "20-30"
    assert projected_bucket(59.99) == "50-60"
    assert projected_bucket(60.0) == ">=60"
    assert projected_bucket(120.0) == ">=60"


# ---------------------------------------------------------------------------
# CLI and window handling
# ---------------------------------------------------------------------------


def test_cli_json_emits_rows_and_summary(tmp_path, capsys):
    log_dir = _write_log(
        tmp_path,
        [
            _admission("2026-10-19 09:00:00,000", session="s-1"),
            _enqueue("2026-10-19 09:00:05,000", session="s-1"),
            _fallback("2026-10-19 09:01:05,000", session="s-1", queued=60.0),
        ],
    )
    assert main(
        [
            "--since",
            "2026-10-19 00:00:00",
            "--until",
            "2026-10-20 00:00:00",
            "--log-dir",
            str(log_dir),
            "--json",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["since"] == "2026-10-19 00:00:00"
    assert report["until"] == "2026-10-20 00:00:00"
    assert len(report["rows"]) == 1
    assert report["summary"]["queue_waits"] == 1
    assert report["log_files"] == ["proxy.log"]


def test_cli_rejects_an_inverted_window(tmp_path, capsys):
    log_dir = _write_log(tmp_path, [])
    assert main(
        [
            "--since",
            "2026-10-20 00:00:00",
            "--until",
            "2026-10-19 00:00:00",
            "--log-dir",
            str(log_dir),
            "--json",
        ]
    ) == 2
    assert "--until must not precede --since" in capsys.readouterr().err


def test_text_output_is_human_readable(tmp_path):
    log_dir = _write_log(
        tmp_path,
        [
            _enqueue("2026-10-19 09:00:05,000", session="s-1"),
            _fallback("2026-10-19 09:01:05,000", session="s-1", queued=60.0),
        ],
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert main(
            [
                "--since",
                "2026-10-19 00:00:00",
                "--until",
                "2026-10-20 00:00:00",
                "--log-dir",
                str(log_dir),
            ]
        ) == 0
    text = buffer.getvalue()

    assert "Queue waits: 1" in text
    assert "Blocker by class" in text


def test_window_filtering_excludes_out_of_window_lines(tmp_path):
    lines = [
        _admission("2026-10-18 09:00:00,000", session="s-1"),
        _enqueue("2026-10-18 09:00:05,000", session="s-1"),
        _fallback("2026-10-18 09:01:05,000", session="s-1", queued=60.0),
    ]
    report = _report(tmp_path, lines)

    assert report["rows"] == []
    assert report["summary"]["queue_waits"] == 0


def test_diagnosis_is_deterministic(tmp_path):
    lines = [
        _lease_prefill("2026-10-19 08:59:00,000", session="holder-a"),
        _admission("2026-10-19 09:00:00,000", session="s-1", new_tokens=6000),
        _enqueue("2026-10-19 09:00:05,000", session="s-1", new_tokens=6000),
        _fallback("2026-10-19 09:01:05,000", session="s-1", queued=60.0, new_tokens=6000),
    ]
    log_dir = _write_log(tmp_path, lines)
    first = analyse(log_dir, SINCE, UNTIL)
    second = analyse(log_dir, SINCE, UNTIL)

    assert first == second


@pytest.mark.parametrize(
    "decision, expected_admits",
    [("admit", 1), ("reject", 0)],
)
def test_guard_decision_is_reported_verbatim(tmp_path, decision, expected_admits):
    lines = [
        _admission("2026-10-19 09:00:00,000", session="s-1", decision=decision),
        _enqueue("2026-10-19 09:00:05,000", session="s-1"),
        _fallback("2026-10-19 09:01:05,000", session="s-1", queued=60.0),
    ]
    summary = _report(tmp_path, lines)["summary"]

    assert summary["abandoned_guard"]["admits"] == expected_admits
    assert summary["abandoned_guard"]["with_guard"] == 1
