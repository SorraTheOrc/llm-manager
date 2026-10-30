#!/usr/bin/env python3
"""Tests for the admission→first-byte conversion harness (LP-0MVS8T6F4006DGG3).

The harness is the pre-change baseline for parent AC4 (queued-local-wait
fallback share vs the 85 % reference). These tests pin the parsing and join
contract from synthetic log lines only — deterministic, no live log
dependency, no sleeps — so the committed baseline numbers stay traceable.

Covered:

- AC1: per-band verdicts, conversion, abandoned-wait counts/shares, queue-wait
  seconds, plus window bounds and the log files read.
- AC2: banding and the admission→first-byte session join; an admit with no
  first byte is unconverted, and a first byte with no admit is reported
  separately rather than silently joined.
- AC3: graceful degradation when the newer log fields (``rate_bucket`` /
  ``contended_samples``) are absent.
- AC4 (post-change record): ``local_eviction_after_acceptance`` events counted
  in total and by ``reason``, window-filtered, never leaking into the
  admission verdict counters, and only when the marker starts the logging
  message (a request body that merely quotes it contributes 0).
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
from prefill_admission_conversion import (
    UNKNOWN_BAND,
    analyse,
    band_for,
    build_report,
    main,
    parse_event,
    parse_line_timestamp,
    parse_timestamp,
)

SINCE = datetime(2026, 10, 19, 0, 0, 0)
UNTIL = datetime(2026, 10, 20, 0, 0, 0)


def _admission(
    ts: str,
    *,
    decision: str = "admit",
    reason: str = "within_budget",
    session: str = "s-1",
    new_tokens: int | None = 6000,
    rate_bucket: str | None = "contended",
    contended_samples: int | None = 12,
) -> str:
    fields = [
        f"local_admission_guard decision={decision} reason={reason}",
        "enforced=false eta_prefill_s=13.9 eta_occupancy_s=16.0",
        "projected_wait_s=29.9 budget_s=60.0",
    ]
    if new_tokens is not None:
        fields.append(f"new_tokens={new_tokens}")
    if rate_bucket is not None:
        fields.append(f"rate_bucket={rate_bucket}")
    if contended_samples is not None:
        fields.append(f"contended_samples={contended_samples}")
    fields.append(f"session={session}")
    return f"{ts} - llama-proxy - INFO - " + " ".join(fields)


def _first_byte(ts: str, session: str = "s-1") -> str:
    return (
        f"{ts} - llama-proxy - INFO - dispatch_first_byte_ms=14000 "
        f"session={session} model=Qwen3 provider=local"
    )


def _queue_fallback(ts: str, session: str, queued: float = 60.0) -> str:
    return (
        f"{ts} - llama-proxy - INFO - contention_queue_fallback_after_queue "
        f"provider=local session={session} queued_duration={queued:.2f}s "
        f"reason=local_lease_active"
    )


def _queue_dispatch(ts: str, session: str, queued: float = 12.0) -> str:
    return (
        f"{ts} - llama-proxy - INFO - contention_queue_dispatch "
        f"provider=local session={session} queued_duration={queued:.2f}s "
        f"policy=queue depth=1"
    )


def _eviction(ts: str, session: str, reason: str | None = "measured_cache_deficit") -> str:
    fields = [
        "local_eviction_after_acceptance",
        f"session={session}",
    ]
    if reason is not None:
        fields.append(f"reason={reason}")
    fields.extend(["estimated_tokens=16788", "new_tokens=81", "eta_prefill_s=15.4"])
    return f"{ts} - ERROR - " + " ".join(fields)


def _write_log(tmp_path: Path, lines: list[str], name: str = "proxy.log") -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / name).write_text("\n".join(lines) + "\n")
    return log_dir


# ---------------------------------------------------------------------------
# AC2 — banding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (0, "<8k"),
        (7_999, "<8k"),
        (8_000, "8k-20k"),
        (19_999, "8k-20k"),
        (20_000, "20k-32k"),
        (31_999, "20k-32k"),
        (32_000, "32k-64k"),
        (63_999, "32k-64k"),
        (64_000, ">=64k"),
        (200_000, ">=64k"),
        (None, UNKNOWN_BAND),
    ],
)
def test_banding_boundaries(tokens, expected):
    assert band_for(tokens) == expected


# ---------------------------------------------------------------------------
# AC1 + AC2 — verdicts, conversion join, abandoned waits
# ---------------------------------------------------------------------------

def _synthetic_events():
    lines = [
        # <8k: s-1 admitted and converted.
        _admission("2026-10-19 10:00:00,000", session="s-1", new_tokens=6000),
        _first_byte("2026-10-19 10:00:14,000", session="s-1"),
        # 8k-20k: s-2 admitted, never converted.
        _admission("2026-10-19 10:01:00,000", session="s-2", new_tokens=10000),
        # 20k-32k: s-3 rejected.
        _admission(
            "2026-10-19 10:02:00,000",
            session="s-3",
            new_tokens=25000,
            decision="reject",
            reason="prefill_eta_exceeds_deadline",
        ),
        # 20k-32k: s-4 queued then abandoned.
        _admission("2026-10-19 10:03:00,000", session="s-4", new_tokens=25000),
        _queue_fallback("2026-10-19 10:04:00,000", "s-4", queued=60.0),
        # 8k-20k: s-5 queued then dispatched.
        _admission("2026-10-19 10:05:00,000", session="s-5", new_tokens=9000),
        _queue_dispatch("2026-10-19 10:05:12,000", "s-5", queued=12.0),
        # First byte with no admission line → reported separately.
        _first_byte("2026-10-19 10:06:00,000", session="ghost"),
    ]
    return [parse_event(line, parse_line_timestamp(line)) for line in lines]


def test_report_counts_verdicts_conversion_and_abandoned_waits():
    events = [event for event in _synthetic_events() if event is not None]
    report = build_report(events, files=["proxy.log"])
    bands = report["bands"]

    assert bands["<8k"]["admission_verdicts"] == {"admit": 1, "reject": 0}
    assert bands["<8k"]["converted"] == 1
    assert bands["<8k"]["unconverted"] == 0
    assert bands["<8k"]["conversion_share"] == 1.0

    assert bands["8k-20k"]["admission_verdicts"] == {"admit": 2, "reject": 0}
    assert bands["8k-20k"]["converted"] == 0
    assert bands["8k-20k"]["unconverted"] == 2
    assert bands["8k-20k"]["queue_dispatched"] == 1
    assert bands["8k-20k"]["queue_abandoned"] == 0
    assert bands["8k-20k"]["queued_waits"] == 1
    assert bands["8k-20k"]["abandoned_share"] == 0.0

    assert bands["20k-32k"]["admission_verdicts"] == {"admit": 1, "reject": 1}
    assert bands["20k-32k"]["reject_reasons"] == {
        "prefill_eta_exceeds_deadline": 1
    }
    assert bands["20k-32k"]["queue_abandoned"] == 1
    assert bands["20k-32k"]["queued_waits"] == 1
    assert bands["20k-32k"]["abandoned_share"] == 1.0
    assert bands["20k-32k"]["queue_wait_seconds"] == 60.0

    # A first byte with no admit is never silently joined to an admission.
    assert report["first_byte_without_admit"] == 1
    assert ">=64k" not in bands


def test_rate_bucket_counts_are_reported_per_band():
    events = _synthetic_events()
    report = build_report([e for e in events if e is not None], files=[])
    assert report["bands"]["<8k"]["rate_buckets"] == {"contended": 1}


# ---------------------------------------------------------------------------
# AC3 — pre-change log shapes (no rate_bucket / contended_samples / new_tokens)
# ---------------------------------------------------------------------------

def test_pre_change_lines_without_newer_fields_still_count():
    """Older lines omit ``rate_bucket``/``contended_samples``; the verdict is
    still counted (rate_bucket degrades to ``unknown``)."""
    line = _admission(
        "2026-10-19 11:00:00,000",
        session="old",
        new_tokens=5000,
        rate_bucket=None,
        contended_samples=None,
    )
    event = parse_event(line, parse_line_timestamp(line))
    assert event is not None
    report = build_report([event], files=[])
    assert report["bands"]["<8k"]["admission_verdicts"]["admit"] == 1
    assert report["bands"]["<8k"]["rate_buckets"] == {"unknown": 1}


def test_lines_without_new_tokens_land_in_unknown_band():
    line = _admission(
        "2026-10-19 11:05:00,000",
        session="ancient",
        new_tokens=None,
        rate_bucket=None,
        contended_samples=None,
    )
    event = parse_event(line, parse_line_timestamp(line))
    assert event is not None
    assert event.new_tokens is None
    report = build_report([event], files=[])
    assert report["bands"][UNKNOWN_BAND]["admission_verdicts"]["admit"] == 1


def test_unrelated_lines_are_ignored():
    ts = datetime(2026, 10, 19, 12, 0, 0)
    assert parse_event("2026-10-19 12:00:00,000 - llama-proxy - INFO - ping", ts) is None


# ---------------------------------------------------------------------------
# AC1 — window bounds, log files read, CLI JSON
# ---------------------------------------------------------------------------

def test_window_filtering_excludes_out_of_window_events(tmp_path):
    log_dir = _write_log(
        tmp_path,
        [
            _admission("2026-10-18 23:59:59,000", session="before", new_tokens=6000),
            _admission("2026-10-19 12:00:00,000", session="inside", new_tokens=6000),
            _admission("2026-10-20 00:00:01,000", session="after", new_tokens=6000),
        ],
    )
    report = analyse(log_dir, SINCE, UNTIL)
    assert report["bands"]["<8k"]["admission_verdicts"]["admit"] == 1
    assert report["since"] == "2026-10-19 00:00:00"
    assert report["until"] == "2026-10-20 00:00:00"
    assert report["log_files"] == ["proxy.log"]


# ---------------------------------------------------------------------------
# AC4 — `local_eviction_after_acceptance` counted by reason (no hand-rolled scan)
# ---------------------------------------------------------------------------

def test_eviction_events_are_counted_by_reason():
    lines = [
        _eviction("2026-10-19 10:00:00,000", "s-1", reason="measured_cache_deficit"),
        _eviction("2026-10-19 10:01:00,000", "s-1", reason="measured_cache_deficit"),
        _eviction("2026-10-19 10:02:00,000", "s-2", reason="prefill_deadline"),
        _eviction("2026-10-19 10:03:00,000", "s-3", reason=None),
    ]
    events = [parse_event(line, parse_line_timestamp(line)) for line in lines]
    report = build_report([event for event in events if event is not None], files=[])

    assert report["evictions"] == {
        "total": 4,
        "by_reason": {
            "measured_cache_deficit": 2,
            "prefill_deadline": 1,
            "unknown": 1,
        },
    }
    # Banded by the evicted request's own new_tokens (81 -> <8k), and the
    # eviction never leaks into the admission verdict counters.
    assert report["bands"]["<8k"]["evictions"] == 4
    assert report["bands"]["<8k"]["admission_verdicts"] == {"admit": 0, "reject": 0}
    assert report["totals"]["evictions"] == 4


def _remote_body_post(ts: str, body_text: str) -> str:
    """A proxied-request log line whose *body* quotes an eviction line."""
    return (
        f"{ts} - INFO - [remote] POST http://192.168.0.199:8000/v1/chat/completions"
        f' -> https://api.example.inference body={{"messages": [{{"role": '
        f'"user", "content": "{body_text}"}}]}}'
    )


def test_request_body_lines_never_count_as_evictions(tmp_path):
    """Body echoes of the marker must contribute 0 (LP-0MVV9JWL9003A6XR).

    The pre-change comparator was inflated 463 vs 152 genuine because any line
    containing the string counted; an agent prompt naming the event (and even
    one quoting a full eviction record with a ``reason=`` field) is not an
    eviction.
    """
    quoted_record = (
        "- ERROR - local_eviction_after_acceptance session=s-9 "
        "reason=prefill_deadline new_tokens=40000"
    )
    lines = [
        _remote_body_post(
            "2026-10-19 10:00:00,000",
            "[test-failure] local_eviction_after_acceptance:prefill_deadline",
        ),
        _remote_body_post("2026-10-19 10:01:00,000", quoted_record),
        _remote_body_post(
            "2026-10-19 10:02:00,000",
            "local_eviction_after_acceptance session=x reason=idle_timeout",
        ),
        _eviction("2026-10-19 10:03:00,000", "s-1", reason="prefill_deadline"),
    ]
    log_dir = _write_log(tmp_path, lines)
    report = analyse(log_dir, SINCE, UNTIL)

    assert report["evictions"] == {
        "total": 1,
        "by_reason": {"prefill_deadline": 1},
    }
    assert report["totals"]["evictions"] == 1
    # The only surviving eviction is the genuine line (new_tokens=81 -> <8k);
    # the body echoes carry no bandable event at all.
    assert report["bands"]["<8k"]["evictions"] == 1
    assert "unknown" not in report["bands"]


def test_eviction_events_are_window_filtered(tmp_path):
    log_dir = _write_log(
        tmp_path,
        [
            _eviction("2026-10-18 23:59:59,000", "before", reason="idle_timeout"),
            _eviction("2026-10-19 12:00:00,000", "inside", reason="idle_timeout"),
            _eviction("2026-10-20 00:00:01,000", "after", reason="idle_timeout"),
        ],
    )
    report = analyse(log_dir, SINCE, UNTIL)
    assert report["evictions"] == {"total": 1, "by_reason": {"idle_timeout": 1}}


def test_cli_json_emits_window_bounds_and_files(tmp_path):
    log_dir = _write_log(
        tmp_path,
        [
            _admission("2026-10-19 10:00:00,000", session="s-1", new_tokens=6659),
            _first_byte("2026-10-19 10:00:14,000", session="s-1"),
        ],
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(
            [
                "--since",
                "2026-10-19 00:00:00",
                "--until",
                "2026-10-20 00:00:00",
                "--log-dir",
                str(log_dir),
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["since"] == "2026-10-19 00:00:00"
    assert payload["until"] == "2026-10-20 00:00:00"
    assert payload["log_files"] == ["proxy.log"]
    assert payload["bands"]["<8k"]["converted"] == 1


def test_until_before_since_is_rejected(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(
            [
                "--since",
                "2026-10-20 00:00:00",
                "--until",
                "2026-10-19 00:00:00",
                "--log-dir",
                str(tmp_path),
                "--json",
            ]
        )
    assert code == 2
