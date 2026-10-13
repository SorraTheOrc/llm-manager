#!/usr/bin/env python3
"""Tests for scripts/prefill_lease_verification.py (LP-0MUXQNPKE008K9MP).

The script is the reproducible measurement behind the live post-deploy
verification of the liveness-based prefill lease extension
(LP-0MUXQ5QL40027VZ4 / LP-0MUXTW4MD003TVIC). These tests run it against a
synthetic proxy log so no production log is needed, and assert the observable
metric contract that the acceptance criteria are written against:

  * AC1 — the liveness-only and in-flight ``lease_extended_during_prefill``
    shapes are counted;
  * AC2 — ``reason=no_progress_watchdog``, ``dispatch_first_byte_timeout`` and
    ``backend_retry ... ReadTimeout`` events are counted;
  * AC3 — long local prefills that complete locally are distinguished from
    short ones and from remote completions.
"""

import gzip
import json
import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "prefill_lease_verification.py",
    )
)

# A synthetic proxy log covering every event class the parser handles.
#   session A: a >180s local prefill (extensions 10:00:00..10:04:00) that
#              finishes locally -> counted as a long local completion.
#   session B: a 90s local prefill (two extensions) -> not long at 180s, but
#              long at a 60s threshold.
#   session C: a local finish with no preceding extension -> not long.
#   session D: a remote finish -> never a local completion.
SYNTH = """\
2026-10-12 10:00:00,000 - INFO - lease_extended_during_prefill session=A progress=None alive=True in_flight=True buffer=30s
2026-10-12 10:00:30,000 - INFO - lease_extended_during_prefill session=A progress=None alive=True in_flight=True buffer=30s
2026-10-12 10:01:00,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:01:30,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:02:00,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:02:30,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:03:00,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:03:30,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:04:00,000 - INFO - lease_extended_during_prefill session=A progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:04:05,000 - INFO - Stream finished: reason=stop tokens=40000/200/40200 session=A provider=local model=Qwen3
2026-10-12 10:09:59,000 - WARNING - prefill_progress_unobservable: no per-slot or aggregate prefill progress; lease extension relies on the adaptive acquisition estimate and liveness signal
2026-10-12 10:10:00,000 - INFO - lease_extended_during_prefill session=B progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:11:30,000 - INFO - lease_extended_during_prefill session=B progress=None alive=False in_flight=True buffer=30s
2026-10-12 10:11:35,000 - INFO - Stream finished: reason=stop tokens=100/5/105 session=B provider=local model=Qwen3
2026-10-12 10:12:00,000 - INFO - Stream finished: reason=stop tokens=50/5/55 session=C provider=local model=Qwen3
2026-10-12 10:13:00,000 - INFO - Stream finished: reason=stop tokens=60/6/66 session=D provider=opencode-go model=deepseek
2026-10-12 10:09:59,500 - WARNING - lease_released session=E reason=no_progress_watchdog endpoint=http://localhost:8080
2026-10-12 10:14:00,000 - WARNING - dispatch_first_byte_timeout session=F ceiling=900s
2026-10-12 10:15:00,000 - WARNING - backend_retry path=v1/chat/completions stream=True attempt=1/8 delay=0.2s signal=timeout_failures error=ReadTimeout
2026-10-12 10:16:00,000 - WARNING - backend_retry path=v1/chat/completions stream=True attempt=2/8 delay=0.4s signal=timeout_failures error=ConnectError
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


def _metrics(log_dir, *extra):
    proc = _run(log_dir, *extra)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    return json.loads(proc.stdout)["metrics"]


def test_lease_extension_shapes_are_counted(tmp_path):
    """AC1: total extensions and the alive/in-flight shapes are measured."""
    m = _metrics(_write_logs(tmp_path))
    assert m["lease_extended_total"] == 11
    assert m["lease_extended_alive_true"] == 2
    assert m["lease_extended_in_flight_true"] == 11


def test_watchdog_and_retry_events_are_counted(tmp_path):
    """AC2: the kill/retry-loop events are measured."""
    m = _metrics(_write_logs(tmp_path))
    assert m["no_progress_watchdog_releases"] == 1
    assert m["dispatch_first_byte_timeout"] == 1
    assert m["backend_retry_read_timeout"] == 1  # ConnectError is not a ReadTimeout
    assert m["prefill_progress_unobservable"] == 1


def test_local_stream_finishes_counted(tmp_path):
    """AC3: only provider=local finishes count as local completions."""
    m = _metrics(_write_logs(tmp_path))
    assert m["local_stream_finishes"] == 3


def test_long_prefill_local_completion_is_distinguished(tmp_path):
    """AC3: a >180s prefill that finishes locally is a long local completion.

    Session A's extensions span four minutes before its local finish; sessions
    B and C finish locally without a long preceding prefill and must not count.
    """
    m = _metrics(_write_logs(tmp_path))
    assert m["long_prefill_local_finishes"] == 1


def test_long_prefill_threshold_is_configurable(tmp_path):
    """AC3: the long-prefill boundary is a parameter, not a hardcoded 180."""
    m = _metrics(_write_logs(tmp_path), "--long-prefill-seconds", "60")
    assert m["long_prefill_local_finishes"] == 2  # session A and session B


def test_window_filtering(tmp_path):
    """Windowing: --since/--until bound the counted corpus."""
    m = _metrics(
        _write_logs(tmp_path),
        "--since", "2026-10-12 10:10:00",
        "--until", "2026-10-12 10:13:00",
    )
    assert m["lease_extended_total"] == 2  # only session B's extensions
    assert m["local_stream_finishes"] == 2  # sessions B and C
    assert m["no_progress_watchdog_releases"] == 0
    assert m["long_prefill_local_finishes"] == 0


def test_reads_gzip_rotated_log(tmp_path):
    """Rotation compatibility: .gz rotated logs are part of the corpus."""
    log_dir = _write_logs(tmp_path, text="")
    with gzip.open(log_dir / "proxy.log-2026-10-11_00.gz", "wt") as fh:
        fh.write(SYNTH)
    m = _metrics(log_dir)
    assert m["lease_extended_total"] == 11


def test_json_meta_is_self_describing(tmp_path):
    """The JSON artifact records the corpus and window it was measured over."""
    data = json.loads(_run(_write_logs(tmp_path)).stdout)
    assert data["meta"]["files"] == ["proxy.log"]
    assert data["meta"]["long_prefill_seconds"] == 180
    assert "generated_at" in data["meta"]


def test_missing_log_dir_exits_nonzero(tmp_path):
    proc = _run(tmp_path / "does-not-exist")
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_human_summary_renders(tmp_path):
    proc = subprocess.run(
        ["python3", SCRIPT, "--log-dir", str(_write_logs(tmp_path))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "lease_extended_during_prefill" in proc.stdout
    assert "no_progress_watchdog" in proc.stdout
    assert "long local prefill completions" in proc.stdout
