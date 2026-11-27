"""Retry-outcome join: abort → the caller's next dispatch (LP-0MWM0CPQN009SHJR).

A post-acceptance abort says an admitted local dispatch was released before its
first byte. It does not say what the caller then experienced — a 15 s abort
followed by an immediate successful retry is cheap, the same abort leaving the
request unanswered is expensive. This module pins both halves of the join:

- **emission** (``proxy.retry_outcome``): a ``local_dispatch_outcome`` line
  carries the correlation key (``session`` + ``prompt_fp``) plus ``outcome`` and
  ``turn``; the eviction line carries the same ``prompt_fp``/``turn``, so the
  analyser can join them;
- **analysis** (``analyse_prefill_deadline_residual.retry_join``): per abort it
  reports the next dispatch outcome (local accepted / local refused / remote /
  none), whether the retry carried the same prompt, the abort→first-byte
  latency, and completion — aggregated per abort reason (AC3).

AC4 is structural: every emission is best-effort and the analysis reads only
log lines, so the diff is logging plus analysis and cannot change an admission
decision, an abort policy or a threshold.
"""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

from proxy import admission_events, retry_outcome

WINDOW_START = datetime(2026, 11, 20, 0, 0, 0)
WINDOW_END = datetime(2026, 11, 20, 1, 0, 0)
LOGGER_NAME = "llama-proxy"


@pytest.fixture(name="script")
def _script():
    """Import the analysis script from ``proxy/scripts`` (no package there)."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("analyse_prefill_deadline_residual")


class _Srv:
    """Minimal server stub exposing only what the emitter/eviction sink touch."""

    def __init__(self):
        self.logger = logging.getLogger(LOGGER_NAME)
        self.local_dispatch_records = {}


def _eviction_line(session, reason, *, elapsed_s, prompt_fp, turn, ts="2026-11-20 00:05:00"):
    return (
        f"{ts},000 - ERROR - local_eviction_after_acceptance session={session} "
        f"reason={reason} estimated_tokens=1000 new_tokens=1000 "
        f"eta_prefill_s=10.0 elapsed_s={elapsed_s} slot_id=1 llama_task_id=unknown "
        f"endpoint=local task_started=no progress_observed=yes queue_wait_s=0.0 "
        f"prompt_fp={prompt_fp} turn={turn}"
    )


def _outcome_line(ts, session, outcome, *, prompt_fp, turn):
    return (
        f"{ts},000 - INFO - local_dispatch_outcome session={session} "
        f"prompt_fp={prompt_fp} outcome={outcome} endpoint=local turn={turn}"
    )


def _local_first_byte(ts, session, *, turn):
    return (
        f"{ts},000 - INFO - dispatch_first_byte_ms=5000.0 "
        f"dispatch_to_first_byte_ms=5000.0 cache_state=cold session={session} "
        f"model=Qwen3 turn={turn}"
    )


def _remote_first_byte(ts, session, *, turn):
    return (
        f"{ts},000 - INFO - ttft_seconds=1.5 session={session} "
        f"provider=openrouter model=Qwen3 turn={turn}"
    )


# ---------------------------------------------------------------------------
# Emission half — the correlation key is on the wire
# ---------------------------------------------------------------------------


def test_dispatch_outcome_line_carries_the_join_key(caplog):
    srv = _Srv()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        retry_outcome.record_dispatch_outcome(
            srv, "sess-1", "fp-abc", retry_outcome.OUTCOME_LOCAL_ACCEPTED,
            endpoint="local",
        )
    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_dispatch_outcome")
    )
    assert "session=sess-1" in line
    assert "prompt_fp=fp-abc" in line
    assert "outcome=local_accepted" in line
    assert "endpoint=local" in line


def test_dispatch_outcome_renders_unknown_without_a_key(caplog):
    srv = _Srv()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        retry_outcome.record_dispatch_outcome(srv, None, None, retry_outcome.OUTCOME_REMOTE)
    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_dispatch_outcome")
    )
    assert "session=unknown" in line
    assert "prompt_fp=unknown" in line
    assert "outcome=remote" in line


def test_dispatch_outcome_never_raises_on_a_broken_server(monkeypatch):
    class _Boom:
        @property
        def logger(self):
            raise RuntimeError("no logger")

    # Must swallow the broken logger, not propagate into a request path.
    assert retry_outcome.record_dispatch_outcome(
        _Boom(), "s", "f", retry_outcome.OUTCOME_LOCAL_REFUSED
    ) is True


def test_prompt_fingerprint_ignores_sampling_parameters():
    a = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.1}
    b = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.9}
    c = {"messages": [{"role": "user", "content": "bye"}]}
    fp_a = retry_outcome.prompt_fingerprint(a)
    assert fp_a == retry_outcome.prompt_fingerprint(b)
    assert fp_a != retry_outcome.prompt_fingerprint(c)
    assert retry_outcome.prompt_fingerprint(None) is None


def test_stamp_prefers_the_request_value_over_a_recompute():
    class _Req:
        pass

    request = _Req()
    stamped = retry_outcome.stamp_prompt_fingerprint(
        request, {"messages": [{"role": "user", "content": "x"}]}
    )
    # Even with a different body, the stamped value wins.
    assert retry_outcome.request_prompt_fingerprint(
        request, {"messages": [{"role": "user", "content": "y"}]}
    ) == stamped


def test_eviction_line_carries_prompt_fp_and_turn(caplog):
    srv = _Srv()
    record = {
        "started_at": 1_000.0,
        "active": True,
        "prompt_fp": "fp-xyz",
        "turn": "ab12cd34",
        "estimated_tokens": 1000,
        "new_tokens": 1000,
        "first_byte_seen": False,
    }
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        emitted = admission_events.record_eviction_after_acceptance(
            srv, "sess-2", admission_events.EVICTION_PREFILL_DEADLINE, record,
        )
    assert emitted is True
    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_eviction_after_acceptance")
    )
    assert "prompt_fp=fp-xyz" in line
    assert "turn=ab12cd34" in line


def test_eviction_line_without_a_fingerprint_is_explicitly_unknown(caplog):
    srv = _Srv()
    record = {"started_at": 1_000.0, "active": True, "first_byte_seen": False}
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        admission_events.record_eviction_after_acceptance(
            srv, "sess-3", admission_events.EVICTION_PREFILL_DEADLINE, record,
        )
    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_eviction_after_acceptance")
    )
    assert "prompt_fp=unknown" in line
    assert "turn=unknown" in line


# ---------------------------------------------------------------------------
# Analysis half — the four outcome cases (AC2/AC5 unit rows)
# ---------------------------------------------------------------------------


def _join(script, lines, hours=1.0):
    log = script.parse_window(list(lines), WINDOW_START, WINDOW_END)
    return log, script.retry_join(log, hours)


def test_abort_then_retry_served_locally(script):
    lines = [
        _eviction_line("u-local", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="fp1", turn="t1"),
        _outcome_line("2026-11-20 00:05:02", "u-local", "local_accepted",
                      prompt_fp="fp1", turn="t2"),
        _local_first_byte("2026-11-20 00:05:07", "u-local", turn="t2"),
    ]
    _, joined = _join(script, lines)
    row = joined["rows"][0]
    assert row["outcome"] == "local_accepted"
    assert row["retried"] is True
    assert row["same_prompt"] is True
    assert row["completed"] is True
    assert row["abort_to_first_byte_s"] == pytest.approx(7.0)


def test_abort_then_retry_served_remotely(script):
    lines = [
        _eviction_line("u-remote", "prefill_stalled", elapsed_s=200.0,
                       prompt_fp="fp2", turn="t1", ts="2026-11-20 00:20:00"),
        _outcome_line("2026-11-20 00:20:03", "u-remote", "remote",
                      prompt_fp="fp2", turn="t2"),
        _remote_first_byte("2026-11-20 00:20:04", "u-remote", turn="t2"),
    ]
    _, joined = _join(script, lines)
    row = joined["rows"][0]
    assert row["outcome"] == "remote"
    assert row["retried"] is True
    assert row["completed"] is True
    assert row["first_byte_provider"] == "openrouter"
    assert row["abort_to_first_byte_s"] == pytest.approx(4.0)


def test_abort_with_no_retry_is_none_and_never_completed(script):
    lines = [
        _eviction_line("u-silent", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="fp3", turn="t1"),
    ]
    _, joined = _join(script, lines)
    row = joined["rows"][0]
    assert row["outcome"] == "none"
    assert row["retried"] is False
    assert row["same_prompt"] is None
    assert row["completed"] is False
    assert joined["total"]["never_completed"] == {"count": 1, "share": 1.0}


def test_retry_refused_is_reported_as_refused_and_never_completed(script):
    lines = [
        _eviction_line("u-refused", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="fp4", turn="t1"),
        _outcome_line("2026-11-20 00:05:01", "u-refused", "local_refused",
                      prompt_fp="fp4", turn="t2"),
    ]
    _, joined = _join(script, lines)
    row = joined["rows"][0]
    assert row["outcome"] == "local_refused"
    assert row["retried"] is True
    assert row["completed"] is False
    assert row["abort_to_first_byte_s"] is None


def test_in_turn_refusal_that_escalates_remotely_collapses_to_remote(script):
    lines = [
        _eviction_line("u-chain", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="fp5", turn="t1"),
        _outcome_line("2026-11-20 00:05:01", "u-chain", "local_refused",
                      prompt_fp="fp5", turn="t2"),
        _outcome_line("2026-11-20 00:05:03", "u-chain", "remote",
                      prompt_fp="fp5", turn="t2"),
        _remote_first_byte("2026-11-20 00:05:06", "u-chain", turn="t2"),
    ]
    _, joined = _join(script, lines)
    row = joined["rows"][0]
    assert row["outcome"] == "remote"
    assert row["completed"] is True
    assert row["abort_to_first_byte_s"] == pytest.approx(6.0)


def test_different_prompt_is_not_the_same_prompt(script):
    lines = [
        _eviction_line("u-diff", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="fpA", turn="t1"),
        _outcome_line("2026-11-20 00:08:00", "u-diff", "local_accepted",
                      prompt_fp="fpB", turn="t2"),
    ]
    _, joined = _join(script, lines)
    assert joined["rows"][0]["same_prompt"] is False


def test_pre_change_window_has_no_fingerprint(script):
    """A pre-change eviction line lacks ``prompt_fp``; the join stays honest."""
    line = (
        "2026-11-20 00:05:00,000 - ERROR - local_eviction_after_acceptance "
        "session=old reason=prefill_deadline estimated_tokens=1000 new_tokens=1000 "
        "eta_prefill_s=unknown elapsed_s=180.0 slot_id=unknown llama_task_id=unknown "
        "endpoint=local"
    )
    _, joined = _join(script, [line])
    row = joined["rows"][0]
    assert row["prompt_fp"] is None
    assert row["same_prompt"] is None
    assert row["outcome"] == "none"


def test_join_does_not_cross_sessions(script):
    lines = [
        _eviction_line("s-A", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="f", turn="t1"),
        _outcome_line("2026-11-20 00:05:02", "s-B", "local_accepted",
                      prompt_fp="f", turn="t2"),
    ]
    _, joined = _join(script, lines)
    assert joined["rows"][0]["outcome"] == "none"


# ---------------------------------------------------------------------------
# The analyser block (AC3 fixture dataset)
# ---------------------------------------------------------------------------


def test_report_publishes_the_retry_join_block(script):
    lines = [
        _eviction_line("w1", "prefill_eta_exceeded", elapsed_s=15.0,
                       prompt_fp="f1", turn="a"),
        _outcome_line("2026-11-20 00:05:02", "w1", "local_accepted",
                      prompt_fp="f1", turn="b"),
        _local_first_byte("2026-11-20 00:05:09", "w1", turn="b"),
        _eviction_line("w2", "prefill_eta_exceeded", elapsed_s=20.0,
                       prompt_fp="f2", turn="c", ts="2026-11-20 00:20:00"),
        _outcome_line("2026-11-20 00:20:01", "w2", "remote",
                      prompt_fp="f2", turn="d"),
        _remote_first_byte("2026-11-20 00:20:03", "w2", turn="d"),
    ]
    log = script.parse_window(lines, WINDOW_START, WINDOW_END)
    report = script.build_report(log, WINDOW_START, WINDOW_END)
    block = report["retry_join"]
    assert block["aborts"] == 2
    assert block["source_fields"] == list(script.RETRY_JOIN_SOURCE_FIELDS)
    assert block["outcome_vocabulary"] == list(script.RETRY_JOIN_OUTCOME_VOCABULARY)
    eta = block["by_reason"]["prefill_eta_exceeded"]
    assert eta["aborts"] == 2
    assert eta["retried"] == {"count": 2, "share": 1.0}
    assert eta["outcome"] == {
        "local_accepted": 1,
        "local_refused": 0,
        "remote": 1,
        "none": 0,
    }
    assert eta["never_completed"] == {"count": 0, "share": 0.0}
    assert eta["abort_to_first_byte_s"]["count"] == 2
    # The tolerance and caveat are published, not left implicit.
    assert "SHA-256" in block["tolerance"]
    assert "cannot correlate across a client restart" in block["caveat"]


def test_empty_retry_join_is_well_formed(script):
    block = script.retry_join(script.WindowLog(), 1.0)
    assert block["aborts"] == 0
    assert block["total"]["never_completed"] == {"count": 0, "share": 0.0}
    assert block["by_reason"] == {}
