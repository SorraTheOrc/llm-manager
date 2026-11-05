#!/usr/bin/env python3
"""Unit tests for ``scripts/probe_sigkill_sender.py``.

The probe names the process that SIGKILLs the supervised backends
(LP-0MW3PLEH8008O4I7). Every test drives the probe's public functions against a
**fake procfs** or a synthetic bpftrace/death-window payload, so nothing here
touches the live host, the live proxy or a real process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_sigkill_sender as probe  # noqa: E402

# ---------------------------------------------------------------------------
# fake procfs helpers
# ---------------------------------------------------------------------------


def _write_proc(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    argv: list[str] | None = None,
    uid: int = 1000,
    gid: int = 1000,
    ppid: int = 1,
    cgroup: str = "/user.slice/user-1000.slice/session-7.scope",
    exe: str | None = None,
) -> Path:
    base = proc_root / str(pid)
    base.mkdir(parents=True)
    (base / "comm").write_text(comm + "\n", encoding="utf-8")
    (base / "cmdline").write_bytes(b"\0".join(a.encode() for a in (argv or [comm])) + b"\0")
    (base / "status").write_text(
        f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t{gid}\t{gid}\t{gid}\t{gid}\n",
        encoding="utf-8",
    )
    # Field 4 of /proc/<pid>/stat is the parent pid; the comm in parens may
    # contain spaces, which is why the parser splits on the last ")".
    (base / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1\n", encoding="utf-8")
    (base / "cgroup").write_text(f"0::/init.scope\n1:name=systemd:{cgroup}\n", encoding="utf-8")
    if exe is not None:
        os.symlink(exe, base / "exe")
    return base


def _set_last_pid(proc_root: Path, pid: int) -> None:
    (proc_root / "loadavg").write_text(f"1.00 1.00 1.00 1/100 {pid}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# parsing / classification
# ---------------------------------------------------------------------------


def test_parse_cmdline_splits_nul_separated_argv():
    assert probe.parse_cmdline(b"fuser\0-k\0008081/tcp\0") == ["fuser", "-k", "8081/tcp"]
    assert probe.parse_cmdline(b"") == []
    # A shell wrapper argv, as recorded for the launcher that spawns the killer.
    assert probe.parse_cmdline(b"bash\0/sandbox/start-proxy.sh\0--restart\0") == [
        "bash",
        "/sandbox/start-proxy.sh",
        "--restart",
    ]


@pytest.mark.parametrize(
    ("comm", "argv"),
    [
        ("xargs", ["xargs", "-r", "kill", "-s", "KILL"]),
        ("fuser", ["fuser", "-k", "8081/tcp"]),
        ("pkill", ["pkill", "-f", "llama-server"]),
        ("bash", ["bash", "-c", "kill -9 1234"]),
        ("python3", ["python3", "-c", "import os; os.kill(1234, 9)"]),
    ],
)
def test_is_kill_like_accepts_signal_senders(comm, argv):
    assert probe.is_kill_like(comm, argv) is True


@pytest.mark.parametrize(
    ("comm", "argv"),
    [
        ("llama-server", ["llama-server", "--port", "8080"]),
        ("tts-server", ["tts-server", "--port", "8081"]),
        ("python3", ["python3", "/tmp/harmless.py"]),
        ("bash", ["bash", "/sandbox/start-proxy.sh"]),
    ],
)
def test_is_kill_like_ignores_non_senders(comm, argv):
    """The victim itself and a plain launcher must never be reported as a sender."""
    assert probe.is_kill_like(comm, argv) is False


def test_parse_bpftrace_line_extracts_sender_and_target():
    parsed = probe.parse_bpftrace_line("xargs pid=2179228 -> tts-server pid=2171100")
    assert parsed == {
        "sender": {"comm": "xargs", "pid": 2179228},
        "target": {"comm": "tts-server", "pid": 2171100},
    }


@pytest.mark.parametrize(
    "line",
    [
        "Attaching 1 probe...",
        "",
        "comm pid=abc -> tts-server pid=12",
        "no arrow here",
    ],
)
def test_parse_bpftrace_line_ignores_noise(line):
    assert probe.parse_bpftrace_line(line) is None


# ---------------------------------------------------------------------------
# procfs snapshots
# ---------------------------------------------------------------------------


def test_snapshot_proc_reads_identity_argv_cgroup_and_parent(tmp_path):
    proc_root = tmp_path / "proc"
    _write_proc(
        proc_root,
        2179228,
        comm="xargs",
        argv=["xargs", "-r", "kill", "-s", "KILL"],
        ppid=2179160,
        cgroup="/user.slice/user-1000.slice/session-7.scope",
        exe="/usr/bin/xargs",
    )
    snapshot = probe.snapshot_proc(2179228, proc_root)
    assert snapshot is not None
    assert snapshot["comm"] == "xargs"
    assert snapshot["argv"] == ["xargs", "-r", "kill", "-s", "KILL"]
    assert snapshot["uid"] == 1000
    assert snapshot["gid"] == 1000
    assert snapshot["ppid"] == 2179160
    assert snapshot["cgroup"] == "/user.slice/user-1000.slice/session-7.scope"
    assert snapshot["exe"] == "/usr/bin/xargs"


def test_snapshot_proc_returns_none_when_the_sender_is_already_gone(tmp_path):
    """A sender that exits between sampling and snapshotting must not crash."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    assert probe.snapshot_proc(424242, proc_root) is None


def test_parse_uid_and_ppid_tolerate_missing_fields():
    assert probe.parse_uid(None) == (None, None)
    assert probe.parse_ppid(None) is None
    assert probe.parse_cgroup(None) is None


# ---------------------------------------------------------------------------
# lookback ring + record format
# ---------------------------------------------------------------------------


def test_sender_ring_window_keeps_only_the_lookback():
    ring = probe.SenderRing()
    ring.add({"pid": 1, "observed_at": 100.0})
    ring.add({"pid": 2, "observed_at": 160.0})
    ring.add({"pid": 3, "observed_at": 190.0})
    assert [ev["pid"] for ev in ring.window(now=195.0, seconds=40)] == [2, 3]
    assert [ev["pid"] for ev in ring.window(now=195.0, seconds=120)] == [1, 2, 3]
    assert ring.window(now=1000.0, seconds=120) == []


def test_sender_ring_is_bounded():
    ring = probe.SenderRing(maxlen=3)
    for pid in range(10):
        ring.add({"pid": pid, "observed_at": float(pid)})
    assert len(ring) == 3
    assert [ev["pid"] for ev in ring.window(now=9.0, seconds=100)] == [7, 8, 9]


def test_build_record_matches_the_death_window_envelope():
    record = probe.build_record(
        label="tts-server",
        victim_pid=2171100,
        detected_at=1793894161.778,
        window_seconds=120,
        senders=[{"comm": "fuser", "pid": 2179241, "uid": 1000}],
        mode="proc",
    )
    # The proxy's death_window record carries these fields; correlation matches
    # on detected_at_unix, so the probe record must expose it identically.
    for key in ("kind", "label", "detected_at", "detected_at_unix", "window_seconds"):
        assert key in record
    assert record["kind"] == "sigkill_sender"
    assert record["detected_at_unix"] == pytest.approx(1793894161.778, abs=1e-3)
    assert record["death"] == {
        "label": "tts-server",
        "victim_pid": 2171100,
        "signal": "SIGKILL",
        "signal_number": 9,
    }
    assert record["sender_count"] == 1
    assert record["senders"][0]["comm"] == "fuser"
    json.loads(json.dumps(record))  # must be serialisable as written to disk


def test_format_record_names_the_sender_or_says_none_was_observed():
    record = probe.build_record(
        label="tts-server",
        victim_pid=2171100,
        detected_at=1793894161.778,
        window_seconds=120,
        senders=[
            {
                "comm": "fuser",
                "pid": 2179241,
                "uid": 1000,
                "cgroup": "/user.slice/user-1000.slice/session-7.scope",
                "ppid": 2179160,
                "argv": ["fuser", "-k", "8081/tcp"],
            }
        ],
        mode="proc",
    )
    line = probe.format_record(record)
    assert "label=tts-server" in line
    assert "senders=1" in line
    assert "comm=fuser pid=2179241 uid=1000" in line
    assert "fuser -k 8081/tcp" in line

    empty = probe.build_record(
        label="tts-server",
        victim_pid=1,
        detected_at=0.0,
        window_seconds=120,
        senders=[],
        mode="proc",
    )
    assert probe.format_record(empty).endswith("sender=<none observed>")


# ---------------------------------------------------------------------------
# capture behaviour
# ---------------------------------------------------------------------------


def test_capture_names_the_sender_through_the_shell_that_spawned_it(tmp_path):
    """A 5 ms ``fuser`` is still attributed to its launcher (uid + cgroup).

    This is the core of the probe: the short-lived helper may be gone before it
    can be snapshotted, but the shell that ran ``start-proxy.sh --restart``
    outlives it, so the record names the launcher's argv, uid and cgroup.
    """
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc(
        proc_root,
        2179160,
        comm="bash",
        argv=["bash", "/tmp/pytest-xyz/proxy/scripts/start-proxy.sh", "--restart"],
        uid=1000,
        gid=1000,
        ppid=2033594,
        cgroup="/user.slice/user-1000.slice/session-7.scope",
    )
    _write_proc(
        proc_root,
        2179241,
        comm="fuser",
        argv=["fuser", "-k", "8081/tcp"],
        ppid=2179160,
    )
    _write_proc(proc_root, 2171100, comm="tts-server", argv=["tts-server", "--port", "8081"])
    _set_last_pid(proc_root, 2179241)

    p = probe.Probe(out_dir=tmp_path / "out", proc_root=proc_root, clock=lambda: 200.0)
    p.capture_new_pids(2179160, 2179241, now=200.0)

    assert len(p.ring) == 1
    sender = p.ring.window(now=200.0, seconds=60)[0]
    assert sender["comm"] == "fuser"
    assert sender["uid"] == 1000
    assert sender["argv"] == ["fuser", "-k", "8081/tcp"]
    assert sender["parent"]["comm"] == "bash"
    assert sender["parent"]["argv"][1].endswith("start-proxy.sh")
    assert sender["parent"]["uid"] == 1000
    assert sender["parent"]["cgroup"] == "/user.slice/user-1000.slice/session-7.scope"


def test_capture_ignores_processes_that_cannot_send_signals(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc(proc_root, 500, comm="llama-server", argv=["llama-server", "--port", "8080"])
    _write_proc(proc_root, 501, comm="python3", argv=["python3", "-m", "pytest"])
    _set_last_pid(proc_root, 501)

    p = probe.Probe(proc_root=proc_root, clock=lambda: 10.0)
    p.capture_new_pids(500, 501, now=10.0)
    assert len(p.ring) == 0


def test_record_death_persists_a_json_record(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc(proc_root, 900, comm="xargs", argv=["xargs", "-r", "kill", "-s", "KILL"])
    out_dir = tmp_path / "out"

    p = probe.Probe(out_dir=out_dir, proc_root=proc_root, clock=lambda: 1000.0)
    p.observe(900, now=1000.0)
    record = p.record_death("tts-server", 950, now=1000.0, mode="proc")

    written = sorted(out_dir.glob("sigkill_sender_tts-server_*.json"))
    assert len(written) == 1
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved["detected_at_unix"] == 1000.0
    assert saved["senders"][0]["pid"] == 900
    assert saved == json.loads(json.dumps(record))


# ---------------------------------------------------------------------------
# correlation against the proxy's death_window record
# ---------------------------------------------------------------------------


def _death_window(detected_at_unix: float) -> dict:
    return {
        "kind": "death_window",
        "label": "tts_server",
        "detected_at": "2026-11-05T15:56:01.747000+00:00",
        "detected_at_unix": detected_at_unix,
        "window_seconds": 120.0,
        "sample_count": 119,
        "samples": [{"timestamp": detected_at_unix - 1.0}],
    }


def test_correlate_matches_only_senders_inside_the_window():
    death = _death_window(1793894161.747)
    events = [
        {"pid": 1, "observed_at": 1793894100.0, "comm": "fuser"},  # 61 s before
        {"pid": 2, "observed_at": 1793894160.3, "comm": "xargs"},  # 1.4 s before
        {"pid": 3, "observed_at": 1793894161.9, "comm": "kill"},  # 0.2 s after
        {"pid": 4, "observed_at": 1793894300.0, "comm": "fuser"},  # far after
    ]
    matched = probe.correlate(death, events, window_seconds=15.0)
    assert [ev["pid"] for ev in matched] == [2, 3]
    assert probe.correlate(death, events, window_seconds=1.0) == [
        {"pid": 3, "observed_at": 1793894161.9, "comm": "kill"}
    ]


def test_correlate_falls_back_to_the_last_sample_timestamp():
    death = {"label": "tts_server", "samples": [{"timestamp": 500.0}, {"timestamp": 520.0}]}
    events = [{"pid": 7, "observed_at": 525.0, "comm": "fuser"}]
    assert [ev["pid"] for ev in probe.correlate(death, events, window_seconds=10.0)] == [7]


def test_correlate_without_any_timestamp_matches_nothing():
    assert probe.correlate({"label": "x"}, [{"pid": 1, "observed_at": 1.0}], 5.0) == []


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_cli_reports_absence_of_evidence_with_exit_code_2(tmp_path, capsys):
    """No death observed => exit 2 (armed, nothing seen) — never "no sender"."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _set_last_pid(proc_root, 10)
    code = probe.main(["--mode", "proc", "--proc-root", str(proc_root), "--rate", "1000", "--duration", "0.05"])
    assert code == 2
    assert "victims=<none>" in capsys.readouterr().out


def test_cli_correlate_mode_prints_the_named_sender(tmp_path, capsys):
    death_path = tmp_path / "death_window_tts_server_example.json"
    death_path.write_text(json.dumps(_death_window(1000.0)), encoding="utf-8")
    events_path = tmp_path / "sender_events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "pid": 2179241,
                "comm": "fuser",
                "uid": 1000,
                "cgroup": "/user.slice/user-1000.slice/session-7.scope",
                "ppid": 2179160,
                "argv": ["fuser", "-k", "8081/tcp"],
                "observed_at": 1000.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code = probe.main(["--mode", "correlate", "--death-window", str(death_path), "--events", str(events_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "candidates=1" in out
    assert "comm=fuser pid=2179241 uid=1000" in out


def test_cli_correlate_mode_reports_no_candidate_with_exit_code_2(tmp_path, capsys):
    death_path = tmp_path / "dw.json"
    death_path.write_text(json.dumps(_death_window(1000.0)), encoding="utf-8")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    code = probe.main(["--mode", "correlate", "--death-window", str(death_path), "--events", str(events_path)])
    assert code == 2
    assert "candidates=0" in capsys.readouterr().out


def test_observe_appends_the_sender_event_log(tmp_path):
    """``--mode correlate`` consumes this JSONL, so observation must write it."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc(proc_root, 700, comm="xargs", argv=["xargs", "-r", "kill", "-s", "KILL"])
    out_dir = tmp_path / "out"

    p = probe.Probe(out_dir=out_dir, proc_root=proc_root, clock=lambda: 42.0)
    p.observe(700, now=42.0)

    events = [json.loads(line) for line in (out_dir / "sender_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["pid"] == 700
    assert events[0]["comm"] == "xargs"
    assert events[0]["observed_at"] == 42.0


def test_json_output_emits_one_record_per_line(tmp_path):
    """`--json` is the machine-readable form of the human record line."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc(proc_root, 800, comm="fuser", argv=["fuser", "-k", "8081/tcp"])
    lines: list[str] = []

    p = probe.Probe(proc_root=proc_root, clock=lambda: 5.0, emit=lines.append, emit_json=True)
    p.observe(800, now=5.0)
    p.record_death("tts-server", 801, now=5.0, mode="proc")

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "sigkill_sender"
    assert record["senders"][0]["pid"] == 800
    assert record["death"]["victim_pid"] == 801
