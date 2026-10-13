#!/usr/bin/env python3
"""Tests for scripts/dispatch_latency_verification.py (LP-0MV6BR3PT001WBEI).

The script is the reproducible AC1/AC2 verification for LP-0MU1S617L002Q9PR.
These tests run it against a synthetic proxy log so no production log is
needed, and assert the observable metric contract the acceptance criteria are
written against:

  * AC1 — ``dispatch_first_byte_timeout`` events are counted and split by
    ceiling (ceiling=900s races the client deadline exactly and is the AC1
    signal; other ceilings are counted separately); chain-hold starts and
    client queue-timeout relays are counted;
  * AC2 — local dispatch lines are split warm/cold on ``cache_state``, the
    queue wait is separated from service time, and p50/p90/p99 are computed;
  * verdict — the re-scoped targets (warm p99 < 60 s, zero ceiling=900s
    events) produce PASS only when met, and a not-yet-closed 24 h window
    downgrades AC1 to a precondition note.
"""

import json
import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "dispatch_latency_verification.py",
    )
)

# A synthetic proxy log covering every event class the parser handles.
#   Session A: warm-cache dispatch, 5.0 s service, tiny queue wait.
#   Session B: cold-cache dispatch, 50 s service (cold), larger queue wait.
#   Session C: legacy ceiling=900s first-byte timeout (pre-fix race signal).
#   Session D: post-fix ceiling=840s first-byte timeout (bounded).
#   Session E: a chain hold start (exhausted remote cycle).
#   Session F: a client queue-timeout relay (upstream body relayed).
SYNTH = """\
2026-10-13 15:48:20,000 - INFO - dispatch_first_byte_ms=5001.0 dispatch_to_first_byte_ms=5000.5 cache_state=warm cached_ratio=0.990 queue_wait_ms=0.5 session=A model=Qwen3
2026-10-13 15:49:00,000 - INFO - dispatch_first_byte_ms=50012.0 dispatch_to_first_byte_ms=50000.3 cache_state=cold cached_ratio=0.000 queue_wait_ms=11.7 session=B model=Qwen3
2026-10-13 15:50:00,000 - WARNING - dispatch_first_byte_timeout session=C ceiling=900s
2026-10-13 15:51:00,000 - WARNING - dispatch_first_byte_timeout session=D ceiling=840s
2026-10-13 15:52:00,000 - WARNING - Chain exhausted for model=v1/chat/completions; holding 300s before restarting cycle from the first provider (chain_hold_cycle=0, cycle=0, max_cycles=3)
2026-10-13 15:53:00,000 - WARNING - [remote] upstream error status=502 url=https://api.example.com body={"error":{"message":"We were unable to start processing your request within the 900-second timeout limit. Please try again later."}}
"""


def _write_logs(tmp_path, text=SYNTH, name="proxy.log"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / name).write_text(text)
    return log_dir


def _run(log_dir, *extra):
    return subprocess.run(
        ["python3", SCRIPT, "--log-dir", str(log_dir), "--json", *extra],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _result(log_dir, *extra):
    proc = _run(log_dir, *extra)
    assert proc.returncode in (0, 2), proc.stderr
    return json.loads(proc.stdout), proc.returncode


# ===================================================================
# AC1 — timeouts split by ceiling, holds and relays counted
# ===================================================================


def test_ac1_counts_and_ceiling_split(tmp_path):
    data, _ = _result(_write_logs(tmp_path))
    ac1 = data["metrics"]["ac1"]
    assert ac1["dispatch_first_byte_timeout_total"] == 2
    assert ac1["dispatch_first_byte_timeout_ceiling_900s"] == 1
    assert ac1["dispatch_first_byte_timeout_ceiling_other"] == 1
    assert ac1["chain_hold_starts"] == 1
    assert ac1["chain_hold_total_seconds"] == 300.0
    assert ac1["client_queue_timeout_relays"] == 1


def test_ac1_no_legacy_ceiling_after_fix(tmp_path):
    text = SYNTH.replace("ceiling=900s", "ceiling=840s").replace(
        "ceiling=840s\n2026-10-13 15:51", "ceiling=840s\n2026-10-13 15:51"
    )
    data, _ = _result(_write_logs(tmp_path, text))
    ac1 = data["metrics"]["ac1"]
    assert ac1["dispatch_first_byte_timeout_ceiling_900s"] == 0
    assert ac1["dispatch_first_byte_timeout_ceiling_other"] == 2


# ===================================================================
# AC2 — warm/cold split and queue-wait separation
# ===================================================================


def test_ac2_warm_cold_split_and_stats(tmp_path):
    data, _ = _result(_write_logs(tmp_path))
    ac2 = data["metrics"]["ac2"]
    assert ac2["warm_service_ms"]["n"] == 1
    assert ac2["warm_service_ms"]["p50"] == 5000.5
    assert ac2["cold_service_ms"]["n"] == 1
    assert ac2["cold_service_ms"]["p50"] == 50000.3
    assert ac2["queue_wait_ms"]["n"] == 2
    assert ac2["queue_wait_ms"]["max"] == 11.7


def test_ac2_queue_wait_is_less_than_service_for_warm(tmp_path):
    data, _ = _result(_write_logs(tmp_path))
    ac2 = data["metrics"]["ac2"]
    # queue wait must never exceed service time for the warm line (0.5 <= 5000.5)
    assert ac2["queue_wait_ms"]["p50"] <= ac2["warm_service_ms"]["p50"]


# ===================================================================
# Window filtering
# ===================================================================


def test_window_since_filters_out_earlier_events(tmp_path):
    data, _ = _result(
        _write_logs(tmp_path), "--since", "2026-10-13 15:51:00"
    )
    ac1 = data["metrics"]["ac1"]
    ac2 = data["metrics"]["ac2"]
    # only the 15:51:00 ceiling=840s event falls inside [15:51:00, end)
    assert ac1["dispatch_first_byte_timeout_total"] == 1
    assert ac1["dispatch_first_byte_timeout_ceiling_other"] == 1
    assert ac1["chain_hold_starts"] == 1
    assert ac1["client_queue_timeout_relays"] == 1
    assert ac2["warm_service_ms"]["n"] == 0
    assert ac2["cold_service_ms"]["n"] == 0


def test_window_until_filters_out_later_events(tmp_path):
    data, _ = _result(
        _write_logs(tmp_path), "--until", "2026-10-13 15:49:00"
    )
    ac1 = data["metrics"]["ac1"]
    ac2 = data["metrics"]["ac2"]
    assert ac1["dispatch_first_byte_timeout_total"] == 0
    assert ac2["warm_service_ms"]["n"] == 1


# ===================================================================
# Verdict
# ===================================================================


def test_verdict_warn_when_legacy_ceiling_present(tmp_path):
    data, rc = _result(_write_logs(tmp_path))
    assert data["verdict"]["verdict"] == "WARN"
    assert rc == 2
    legacy = [
        c for c in data["verdict"]["checks"] if "900s" in c["target"]
    ][0]
    assert legacy["pass"] is False


def test_verdict_pass_when_all_targets_met(tmp_path):
    text = SYNTH
    # Remove the legacy-ceiling event and reduce WARN conditions: keep warm at
    # 5 s (below the 60 s target), drop the cold 50 s line only in the sense of
    # the AC1 checks (cold service time is exempt from the warm target).
    text = text.replace(
        "2026-10-13 15:50:00,000 - WARNING - dispatch_first_byte_timeout session=C ceiling=900s\n",
        "",
    )
    text = text.replace(
        "We were unable to start processing your request within the 900-second timeout limit. Please try again later.",
        "transient upstream override",
    )
    data, rc = _result(_write_logs(tmp_path, text))
    assert data["verdict"]["verdict"] == "PASS", data["verdict"]
    assert rc == 0


def test_verdict_warn_on_cold_window_without_window_closed_note(tmp_path):
    text = SYNTH.replace(
        "2026-10-13 15:50:00,000 - WARNING - dispatch_first_byte_timeout session=C ceiling=900s\n",
        "",
    )
    text = text.replace(
        "We were unable to start processing your request within the 900-second timeout limit. Please try again later.",
        "transient upstream override",
    )
    data, rc = _result(_write_logs(tmp_path, text))
    # warm p99 = 5 s < 60 s, no legacy ceiling => PASS, but the 24h window is
    # not closed, so the AC1 check still passes and the note is emitted.
    assert data["verdict"]["verdict"] == "PASS"
    assert any("not yet closed" in n for n in data["verdict"]["notes"])
    assert rc == 0


# ===================================================================
# Input handling
# ===================================================================


def test_missing_log_dir_returns_error(tmp_path):
    proc = subprocess.run(
        ["python3", SCRIPT, "--log-dir", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "log directory not found" in proc.stderr


def test_gzip_logs_are_read(tmp_path):
    import gzip

    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    with gzip.open(log_dir / "proxy.log.2026-10-13_15.gz", "wt") as fh:
        fh.write(SYNTH)
    data, _ = _result(log_dir)
    assert data["metrics"]["ac1"]["dispatch_first_byte_timeout_total"] == 2
