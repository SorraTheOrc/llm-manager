#!/usr/bin/env python3
"""Name the process that delivers ``SIGKILL`` to the supervised LLM backends.

Why this exists
---------------
Class A deaths (``SIGKILL(9)/137`` of the ``llama-server`` router and, seconds
later, of ``tts-server``) are invisible without help: ``SIGKILL`` cannot be
caught, the kernel logs nothing for it, and this host blocks every unprivileged
observation of signal delivery (``ptrace_scope=1``,
``perf_event_paranoid=4``, ``unprivileged_bpf_disabled=2``, no passwordless
``sudo``).  ``docs/death-attribution-LP-0MW3K7WYV001XLL1.md`` establishes all of
that and hands the identification of the sender to
``LP-0MW3PLEH8008O4I7``.

Two modes, one record format
----------------------------
``--mode proc`` (default, unprivileged)
    Watches the victims (``llama-server`` / ``tts-server``) and the host's pid
    churn at ``--rate`` Hz, snapshotting every newly forked process while it is
    still alive: ``comm``, argv, ``exe``, ``uid``/``gid``, ``cgroup`` and
    ``ppid`` — plus the parent's snapshot, so a sender that lives only a few
    milliseconds is still named through the shell that spawned it.  When a
    victim disappears, the candidate senders observed in the lookback window
    are written as a ``sigkill_sender`` record.

    This mode *correlates*: it names processes that were running and look like
    signal senders at the moment of a death.  It cannot prove the signal came
    from them (nothing unprivileged can), which is why the evidence is quoted
    together with the victim's own death record and its proxy line.

``--mode bpftrace`` (needs ``sudo``)
    Runs ``bpftrace`` on ``tracepoint:signal:signal_generate`` filtered to
    ``sig == 9`` for a bounded ``--duration`` and records the **exact** sender
    (``comm``/``pid``) of every kill, enriched with a ``/proc`` snapshot when
    the sender is still alive.  This is the authoritative attribution.

``--mode correlate``
    Cross-checks a ``death_window_*.json`` written by
    ``proxy/proxy/death_capture.py`` against a probe event log, printing the
    candidate senders whose observation falls inside
    ``detected_at ± --correlate-window``.

Output format
-------------
One JSON record per death, in the same envelope as the proxy's
``death_window`` record (``kind`` / ``label`` / ``detected_at`` /
``detected_at_unix`` / ``window_seconds`` / ``samples``) so the two can be
correlated by timestamp, plus the attribution-specific ``senders`` list.
``--out-dir`` persists them as ``sigkill_sender_<label>_<stamp>.json``.

Exit codes: ``0`` at least one death window was captured; ``2`` the probe ran
and observed nothing (absence of evidence — leave it armed and re-run, never
conclude "no sender"); ``1`` usage/IO error.

Usage
-----
    # unprivileged correlation probe, 30 minutes, written next to the proxy log
    python3 scripts/probe_sigkill_sender.py --duration 1800 \
        --out-dir /var/log/llama-proxy

    # same, machine-readable on stdout (one JSON record per line)
    python3 scripts/probe_sigkill_sender.py --duration 1800 --json \
        --out-dir /var/log/llama-proxy

    # authoritative attribution (bounded; needs operator sudo)
    sudo python3 scripts/probe_sigkill_sender.py --mode bpftrace --duration 3600 \
        --out-dir /var/log/llama-proxy

    # cross-check one captured death against a probe event log
    python3 scripts/probe_sigkill_sender.py --mode correlate \
        --death-window /var/log/llama-proxy/death_window_tts_server_20261105T155601Z.json \
        --events /var/log/llama-proxy/sender_events.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

#: Process names of the supervised victims (router + workers + TTS).
DEFAULT_VICTIMS = ("llama-server", "tts-server", "qwentts-server")

#: ``comm`` values of the helpers a kill can be funnelled through.
KILL_HELPER_COMMS = frozenset({"kill", "killall", "killall5", "pkill", "xargs", "fuser"})

#: argv fragments that mark a command line as a signal sender, in addition to
#: :data:`KILL_HELPER_COMMS`.
KILL_ARGV_MARKERS = (
    "kill -s ",
    "kill -9",
    "kill -KILL",
    "pkill",
    "killall",
    "fuser -k",
    "os.kill(",
    "os.killpg(",
)

#: Human-facing name for each signal, used only in the record.
SIGNAL_NAMES = {9: "SIGKILL", 15: "SIGTERM"}

#: bpftrace program: one line per ``SIGKILL`` delivery.
BPFTRACE_PROGRAM = (
    "tracepoint:signal:signal_generate /args->sig == 9 / "
    '{ printf("%s pid=%d -> %s pid=%d\\n", comm, pid, args->comm, args->pid); }'
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_probe_sigkill_sender.py)
# ---------------------------------------------------------------------------


def parse_cmdline(raw: bytes | str) -> list[str]:
    """Split a ``/proc/<pid>/cmdline`` payload into argv tokens."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogateescape")
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def is_kill_like(comm: str, argv: Sequence[str]) -> bool:
    """True when *comm*/*argv* look like a process that sends signals."""
    if comm.strip() in KILL_HELPER_COMMS:
        return True
    command = " ".join(argv)
    return any(marker in command for marker in KILL_ARGV_MARKERS)


def parse_bpftrace_line(line: str) -> dict | None:
    """Parse ``<comm> pid=<pid> -> <comm> pid=<pid>`` (the probe's printf).

    Returns ``{"sender": {"comm", "pid"}, "target": {"comm", "pid"}}`` or
    ``None`` for anything else (bpftrace banners, ``Attaching …`` notices).
    """
    line = line.strip()
    if "->" not in line:
        return None
    left, _, right = line.partition("->")
    try:
        sender_comm, sender_pid = left.rsplit("pid=", 1)
        target_comm, target_pid = right.rsplit("pid=", 1)
        return {
            "sender": {"comm": sender_comm.strip(), "pid": int(sender_pid)},
            "target": {"comm": target_comm.strip(), "pid": int(target_pid)},
        }
    except (ValueError, IndexError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def parse_uid(status_text: str | None) -> tuple[int | None, int | None]:
    """Return ``(uid, gid)`` from ``/proc/<pid>/status`` text."""
    uid = gid = None
    for line in (status_text or "").splitlines():
        if line.startswith("Uid:") and uid is None:
            fields = line.split()
            if len(fields) > 1 and fields[1].isdigit():
                uid = int(fields[1])
        elif line.startswith("Gid:") and gid is None:
            fields = line.split()
            if len(fields) > 1 and fields[1].isdigit():
                gid = int(fields[1])
    return uid, gid


def parse_ppid(stat_text: str | None) -> int | None:
    """Return the parent pid from ``/proc/<pid>/stat`` text."""
    if not stat_text or ")" not in stat_text:
        return None
    tail = stat_text.rsplit(")", 1)[1].split()
    if len(tail) < 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


def parse_cgroup(cgroup_text: str | None) -> str | None:
    """Return the cgroup path (last ``cgroup`` line) of a process."""
    lines = [ln for ln in (cgroup_text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1].split(":", 2)[-1].strip() or None


def snapshot_proc(pid: int, proc_root: Path | str = "/proc") -> dict | None:
    """Snapshot a live process, or ``None`` when it no longer exists."""
    root = Path(proc_root)
    base = root / str(pid)
    comm = _read_text(base / "comm")
    if comm is None:
        return None
    uid, gid = parse_uid(_read_text(base / "status"))
    exe: str | None = None
    try:
        exe = os.readlink(base / "exe")
    except OSError:
        exe = None
    return {
        "pid": pid,
        "comm": comm.strip(),
        "argv": parse_cmdline(_read_bytes(base / "cmdline") or b""),
        "exe": exe,
        "uid": uid,
        "gid": gid,
        "cgroup": parse_cgroup(_read_text(base / "cgroup")),
        "ppid": parse_ppid(_read_text(base / "stat")),
    }


def last_allocated_pid(proc_root: Path | str = "/proc") -> int | None:
    """The kernel's most recently allocated pid (``/proc/loadavg`` field 5)."""
    text = _read_text(Path(proc_root) / "loadavg")
    if not text:
        return None
    fields = text.split()
    try:
        return int(float(fields[4]))
    except (IndexError, ValueError):
        return None


def discover_victims(names: Iterable[str] = DEFAULT_VICTIMS, proc_root: Path | str = "/proc") -> dict[int, str]:
    """``{pid: comm}`` for every live process whose ``comm`` is a victim name."""
    wanted = set(names)
    found: dict[int, str] = {}
    root = Path(proc_root)
    try:
        entries = os.listdir(root)
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        comm = _read_text(root / entry / "comm")
        if comm is not None and comm.strip() in wanted:
            found[int(entry)] = comm.strip()
    return found


def pid_alive(pid: int, proc_root: Path | str = "/proc") -> bool:
    """True while ``/proc/<pid>`` still exists."""
    return Path(proc_root, str(pid)).exists()


class SenderRing:
    """Bounded lookback buffer of observed candidate senders."""

    def __init__(self, maxlen: int = 512) -> None:
        self._events: deque[dict] = deque(maxlen=maxlen)

    def add(self, event: Mapping) -> None:
        self._events.append(dict(event))

    def __len__(self) -> int:
        return len(self._events)

    def window(self, now: float, seconds: float) -> list[dict]:
        """Events observed at or after ``now - seconds``, oldest first."""
        cutoff = float(now) - float(seconds)
        return [ev for ev in self._events if float(ev.get("observed_at", 0)) >= cutoff]


def build_record(
    *,
    label: str,
    victim_pid: int,
    detected_at: float,
    window_seconds: float,
    senders: Sequence[Mapping],
    mode: str,
    extra: Mapping | None = None,
) -> dict:
    """Build a ``sigkill_sender`` record in the proxy's death-window envelope."""
    record = {
        "kind": "sigkill_sender",
        "label": label,
        "mode": mode,
        "detected_at": datetime.fromtimestamp(detected_at, tz=UTC).isoformat(),
        "detected_at_unix": round(float(detected_at), 3),
        "window_seconds": round(float(window_seconds), 3),
        "death": {
            "label": label,
            "victim_pid": int(victim_pid),
            "signal": SIGNAL_NAMES[9],
            "signal_number": 9,
        },
        "sender_count": len(senders),
        "senders": [dict(sender) for sender in senders],
    }
    if extra is not None:
        record["extra"] = dict(extra)
    return record


def format_sender(sender: Mapping) -> str:
    """One-line human summary of a candidate sender."""
    argv = " ".join(sender.get("argv") or []) or "<no argv>"
    if len(argv) > 160:
        argv = argv[:157] + "..."
    return (
        f"comm={sender.get('comm')} pid={sender.get('pid')} "
        f"uid={sender.get('uid')} cgroup={sender.get('cgroup')} "
        f"ppid={sender.get('ppid')} argv={argv!r}"
    )


def format_record(record: Mapping) -> str:
    """One compact log line for a probe record (mirrors ``format_record`` in
    ``proxy/proxy/death_capture.py``)."""
    death = record.get("death") or {}
    senders = record.get("senders") or []
    head = (
        f"label={record.get('label')} victim_pid={death.get('victim_pid')} "
        f"mode={record.get('mode')} window={record.get('window_seconds')}s "
        f"senders={record.get('sender_count')}"
    )
    if not senders:
        return head + " sender=<none observed>"
    return head + " " + " | ".join(format_sender(s) for s in senders)


def correlate(death_window: Mapping, events: Iterable[Mapping], window_seconds: float) -> list[dict]:
    """Candidate senders observed around a ``death_window`` record.

    Matches on the proxy record's ``detected_at_unix`` (falling back to the
    latest sample timestamp), within ``±window_seconds``.
    """
    reference = death_window.get("detected_at_unix")
    if reference is None:
        samples = death_window.get("samples") or []
        stamps = [s.get("timestamp") for s in samples if s.get("timestamp")]
        reference = max(stamps) if stamps else None
    if reference is None:
        return []
    reference = float(reference)
    matched = []
    for event in events:
        observed = event.get("observed_at")
        if observed is None:
            continue
        if abs(float(observed) - reference) <= float(window_seconds):
            matched.append(dict(event))
    matched.sort(key=lambda ev: float(ev["observed_at"]))
    return matched


# ---------------------------------------------------------------------------
# Probe runtime
# ---------------------------------------------------------------------------


def _iso(unix_ts: float) -> str:
    return datetime.fromtimestamp(unix_ts, tz=UTC).strftime("%Y%m%dT%H%M%SZ")


class Probe:
    def __init__(
        self,
        *,
        out_dir: Path | None = None,
        victim_names: Sequence[str] = DEFAULT_VICTIMS,
        window_seconds: float = 120.0,
        proc_root: Path | str = "/proc",
        clock: Callable[[], float] = time.time,
        emit: Callable[[str], None] = print,
        emit_json: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.victim_names = tuple(victim_names)
        self.window_seconds = float(window_seconds)
        self.proc_root = Path(proc_root)
        self.clock = clock
        self.emit = emit
        self.emit_json = emit_json
        self.ring = SenderRing()
        self.records: list[dict] = []
        self._parent_cache: dict[int, dict] = {}
        self.event_log: Path | None = (self.out_dir / "sender_events.jsonl") if self.out_dir is not None else None

    # -- candidate capture ---------------------------------------------------
    def observe(self, pid: int, now: float) -> dict | None:
        """Snapshot *pid*; remember and return it when it looks like a sender."""
        info = snapshot_proc(pid, self.proc_root)
        if info is None:
            return None
        if not is_kill_like(info["comm"], info["argv"]):
            return None
        event = dict(info)
        event["observed_at"] = round(float(now), 3)
        parent_pid = info.get("ppid")
        if isinstance(parent_pid, int):
            cached = self._parent_cache.get(parent_pid)
            if cached is None:
                cached = snapshot_proc(parent_pid, self.proc_root) or {}
                self._parent_cache[parent_pid] = cached
            if cached:
                event["parent"] = {key: cached.get(key) for key in ("pid", "comm", "argv", "uid", "cgroup")}
        self.ring.add(event)
        self._append_event(event)
        return event

    def _append_event(self, event: Mapping) -> None:
        """Append ``event`` to the JSONL sender log consumed by ``--mode correlate``."""
        if self.event_log is None:
            return
        try:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"probe: cannot append to {self.event_log}: {exc}", file=sys.stderr)

    def capture_new_pids(self, previous_last: int, current_last: int, now: float) -> dict[int, str]:
        """Observe every pid allocated in ``(previous_last, current_last]``.

        Returns ``{pid: comm}`` for newly spawned victims.
        """
        spawned_victims: dict[int, str] = {}
        if current_last < previous_last:
            return spawned_victims
        for pid in range(previous_last + 1, current_last + 1):
            event = self.observe(pid, now)
            comm_text = _read_text(self.proc_root / str(pid) / "comm")
            if comm_text and comm_text.strip() in self.victim_names:
                spawned_victims[pid] = comm_text.strip()
            _ = event
        return spawned_victims

    # -- records -------------------------------------------------------------
    def record_death(self, label: str, victim_pid: int, now: float, mode: str) -> dict:
        senders = self.ring.window(now, self.window_seconds)
        record = build_record(
            label=label,
            victim_pid=victim_pid,
            detected_at=now,
            window_seconds=self.window_seconds,
            senders=senders,
            mode=mode,
        )
        self.records.append(record)
        self.emit(json.dumps(record, sort_keys=True) if self.emit_json else format_record(record))
        self._persist(record)
        return record

    def _persist(self, record: Mapping) -> Path | None:
        if self.out_dir is None:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            target = self.out_dir / (
                f"sigkill_sender_{record.get('label', 'unknown')}_{_iso(float(record['detected_at_unix']))}.json"
            )
            target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"probe: cannot write record: {exc}", file=sys.stderr)
            return None
        return target


def run_proc_mode(probe: Probe, *, rate: float, deadline: float | None, once: bool) -> int:
    """Sample pid churn and victim liveness; emit a record per victim death."""
    interval = 1.0 / max(rate, 1.0)
    victims = discover_victims(probe.victim_names, probe.proc_root)
    previous_last = last_allocated_pid(probe.proc_root) or 0
    probe.emit(
        f"probe: mode=proc rate={rate}Hz window={probe.window_seconds}s victims={sorted(victims.values()) or '<none>'}"
    )
    while deadline is None or probe.clock() < deadline:
        time.sleep(interval)
        now = probe.clock()
        current_last = last_allocated_pid(probe.proc_root)
        if current_last is not None:
            spawned = probe.capture_new_pids(previous_last, current_last, now)
            victims.update(spawned)
            previous_last = max(previous_last, current_last)
        for pid, comm in list(victims.items()):
            if pid_alive(pid, probe.proc_root):
                continue
            del victims[pid]
            probe.record_death(comm, pid, now, "proc")
            if once:
                return 0
        if probe._parent_cache and len(probe._parent_cache) > 1024:
            probe._parent_cache.clear()
    return 0 if probe.records else 2


def run_bpftrace_mode(probe: Probe, *, duration: float, once: bool, sudo: str) -> int:
    """Run the root probe and record the exact sender of every ``SIGKILL``."""
    if shutil.which("bpftrace") is None:
        print("probe: bpftrace is not installed; use --mode proc", file=sys.stderr)
        return 1
    argv = [*([sudo] if sudo else []), "bpftrace", "-e", BPFTRACE_PROGRAM]
    deadline = time.time() + duration
    probe.emit(f"probe: mode=bpftrace duration={duration}s cmd={' '.join(argv)}")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"probe: cannot start bpftrace: {exc}", file=sys.stderr)
        return 1
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.time()
            parsed = parse_bpftrace_line(line)
            if parsed is None:
                print(f"bpftrace: {line.rstrip()}", file=sys.stderr)
                continue
            if parsed["target"]["comm"] not in probe.victim_names:
                continue
            sender_pid = parsed["sender"]["pid"]
            event = snapshot_proc(sender_pid, probe.proc_root) or {
                "pid": sender_pid,
                "comm": parsed["sender"]["comm"],
                "argv": [],
                "uid": None,
                "gid": None,
                "cgroup": None,
                "ppid": None,
            }
            event["observed_at"] = round(now, 3)
            event["attribution"] = "bpftrace:signal_generate"
            probe.record_death(parsed["target"]["comm"], parsed["target"]["pid"], now, "bpftrace")
            probe.records[-1]["senders"] = [event]
            probe.records[-1]["sender_count"] = 1
            if once:
                return 0
            if time.time() >= deadline:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - hardening
            proc.kill()
    return 0 if probe.records else 2


def run_correlate_mode(*, death_window_path: Path, events_path: Path, window: float) -> int:
    """Print candidate senders observed around one captured death."""
    try:
        death_window = json.loads(death_window_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"probe: cannot read {death_window_path}: {exc}", file=sys.stderr)
        return 1
    events = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    except (OSError, ValueError) as exc:
        print(f"probe: cannot read {events_path}: {exc}", file=sys.stderr)
        return 1
    matched = correlate(death_window, events, window)
    print(f"death_window={death_window_path} candidates={len(matched)}")
    for event in matched:
        print("  " + format_sender(event))
    return 0 if matched else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Name the process that SIGKILLs the supervised llama-server/tts-server "
            "(class A backend deaths). See docs/death-attribution-class-a-"
            "LP-0MW3PLEH8008O4I7.md."
        )
    )
    parser.add_argument("--mode", choices=("proc", "bpftrace", "correlate"), default="proc")
    parser.add_argument("--duration", type=float, default=0.0, help="bounded run, seconds (0 = until interrupted)")
    parser.add_argument("--rate", type=float, default=50.0, help="pid-churn samples per second (proc mode)")
    parser.add_argument("--window", type=float, default=120.0, help="lookback seconds kept with each death")
    parser.add_argument("--victim", action="append", default=None, help="victim comm (repeatable)")
    parser.add_argument("--out-dir", default=None, help="directory for the JSON records and event log")
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--once", action="store_true", help="stop after the first death record")
    parser.add_argument("--json", action="store_true", help="print each record as one JSON line")
    parser.add_argument("--sudo", default="sudo", help="privilege prefix for bpftrace mode ('' for none)")
    parser.add_argument("--death-window", default=None, help="correlate mode: death_window_*.json")
    parser.add_argument("--events", default=None, help="correlate mode: probe event log (JSONL)")
    parser.add_argument("--correlate-window", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.mode == "correlate":
        if not args.death_window or not args.events:
            print("probe: --death-window and --events are required", file=sys.stderr)
            return 1
        return run_correlate_mode(
            death_window_path=Path(args.death_window),
            events_path=Path(args.events),
            window=args.correlate_window,
        )

    probe = Probe(
        out_dir=out_dir,
        victim_names=tuple(args.victim) if args.victim else DEFAULT_VICTIMS,
        window_seconds=args.window,
        proc_root=args.proc_root,
        emit_json=args.json,
    )
    deadline = time.time() + args.duration if args.duration > 0 else None
    if args.mode == "proc":
        return run_proc_mode(probe, rate=args.rate, deadline=deadline, once=args.once)
    return run_bpftrace_mode(probe, duration=args.duration or 600.0, once=args.once, sudo=args.sudo)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
