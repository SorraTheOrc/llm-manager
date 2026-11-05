
# <!-- REFACTOR-LP-0MW3L8JJF008R5NB
# smell: modernization
# severity: medium
# description: Use format specifiers instead of percent format
# -->
"""True termination-cause decoding and second-resolution death capture.

Motivation (LP-0MW3K7WD6003TJKZ, parent LP-0MW2RI9AM007O1FO)
-----------------------------------------------------------

The proxy used to report a blanket ``llama_server_exit_code_137`` for every
backend death — 133 times across the retained log rotations — while
``/var/log/kern.log`` recorded ``SIGSEGV`` / general-protection faults for the
*same* processes. The TTS watchdog only logged ``code=-9``. One integer hides
two different facts:

* **which signal** ended the process — ``137`` is ``128 + SIGKILL``, but the
  same process can die of ``SIGSEGV`` (139) or ``SIGTERM`` (143) instead;
* **who delivered it** — the proxy's own process-group ``SIGKILL`` during a
  restart looks exactly like an external kill once it is serialised into
  ``137``.

Two halves fix that.

Decoding the cause (:func:`decode_wait_status`, :func:`decode_returncode`,
:func:`peek_death_status`, :func:`probe_death`)
    A :class:`DeathStatus` names the cause: ``kind``
    (``signal``/``exit``/``unknown``), the shell-convention ``code``
    (``128 + signal`` for a signal death) and the signal name, so a restart
    reason reads ``llama_server_signal_SIGSEGV`` instead of
    ``llama_server_exit_code_137``. ``probe_death`` prefers the **raw** wait
    status read with ``waitid(WNOWAIT)`` *before* the ``Popen`` handle reaps
    the child, because ``Popen.poll()`` collapses a wrapper's signal death into
    bash's ``128 + N`` exit code. ``decode_wait_status`` is the strict POSIX
    decoder for such raw statuses; ``decode_returncode`` handles a
    ``Popen`` return code (``None`` while alive, negative for a signal death,
    ``128 + N`` for a shell wrapper).

The death-window instrument (:class:`DeathCapture`)
    A bounded ring of second-resolution samples — GPU GTT, ``MemAvailable``,
    per-process RSS and ``oom_score_adj``, cgroup ``memory.events``. When a
    death is observed the ring is dumped as one compact ``death_window``
    record: the last N seconds *before* the death, which is the evidence the
    attribution work needs and which nothing on the host retains today.
    Arming is config-gated (``server.death_capture_enabled``) and every read is
    best-effort: an unreadable ``/proc`` file or a missing sysfs attribute
    degrades to ``None`` instead of raising.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Defaults (overridable through the ``server:`` config block)
# ---------------------------------------------------------------------------

#: Sampling interval for the ring buffer, in seconds. One sample per second is
#: the resolution the attribution work needs; the ring holds a two-minute
#: window, so a death record costs a few hundred kilobytes at most.
DEFAULT_INTERVAL_SECONDS = 1.0

#: How far back a death record reaches.
DEFAULT_WINDOW_SECONDS = 120.0

#: Hard cap on the number of retained samples (``window / interval`` plus
#: slack). Bounds memory even when a caller samples much faster than 1 Hz.
DEFAULT_CAPACITY = 240

#: Samples emitted in one death record. The window is bounded by *count* as
#: well as by time so a fast sampler cannot flood the log line.
DEFAULT_MAX_RECORD_SAMPLES = 60

DEFAULT_CGROUP_MEMORY_EVENTS = Path("/sys/fs/cgroup/memory.events")

#: Highest signal number considered when decoding the shell ``128 + N``
#: convention (Linux caps at 64).
MAX_SIGNAL_NUMBER = 64

_SIGNAL_NUMBERS = frozenset(int(sig) for sig in signal.Signals)
_CLD_DUMPED = getattr(os, "CLD_DUMPED", -1)


# ---------------------------------------------------------------------------
# Termination cause
# ---------------------------------------------------------------------------


def signal_name(signum) -> str:
    """Name for *signum* (``"SIGSEGV"``), or ``"SIG<n>"`` when unknown."""
    try:
        return signal.Signals(int(signum)).name
    except (TypeError, ValueError):
        return "SIG%s" % signum


class DeathStatus(NamedTuple):
    """How a watched process ended.

    ``kind`` is ``"signal"`` (ended by a signal, ``code`` is ``128 + signal``),
    ``"exit"`` (a normal exit, ``code`` is the exit status) or ``"unknown"``
    (the status could not be decoded — never reported as a cause).
    ``signal_name`` is the signal name for a signal death and ``None``
    otherwise.
    """

    kind: str
    code: int | None
    signal_name: str | None

    @property
    def is_signal(self) -> bool:
        return self.kind == "signal"

    @property
    def signal_number(self) -> int | None:
        """The signal number for a signal death, else ``None``."""
        if self.kind != "signal" or self.code is None:
            return None
        return int(self.code) - 128

    def describe(self) -> str:
        """Human/log form: ``SIGSEGV(11)/139`` or ``exit code 3``."""
        if self.kind == "signal":
            return "%s(%s)/%s" % (self.signal_name, self.signal_number, self.code)
        if self.kind == "exit":
            return "exit code %s" % self.code
        return "unknown cause"

    def restart_reason(self, label: str) -> str:
        """``restart_reason`` vocabulary value for this cause.

        ``llama_server_signal_SIGSEGV`` / ``tts_server_signal_SIGKILL`` for a
        signal death and ``llama_server_exit_code_3`` for a normal exit, so a
        restart is never anonymous and never a blanket ``137``.
        """
        if self.kind == "signal":
            return "%s_signal_%s" % (label, self.signal_name)
        if self.kind == "exit":
            return "%s_exit_code_%s" % (label, self.code)
        return "%s_unknown_exit" % label

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "code": self.code,
            "signal_name": self.signal_name,
            "signal_number": self.signal_number,
            "describe": self.describe(),
        }


def _signal_death(signum: int) -> DeathStatus:
    signum = int(signum)
    return DeathStatus("signal", 128 + signum, signal_name(signum))


_UNKNOWN = DeathStatus("unknown", None, None)


def decode_wait_status(status) -> DeathStatus:
    """Decode a **raw POSIX wait status** (``waitid``/``wait`` style).

    Uses the kernel's own macros (``os.WIFSIGNALED``/``os.WTERMSIG``), so a
    synthetic ``SIGSEGV`` status decodes to ``SIGSEGV(11)/139`` and is never
    mistaken for its ``128 + N`` exit-code shadow:

    >>> decode_wait_status(signal.SIGSEGV)          # raw signal-death status
    DeathStatus(kind='signal', code=139, signal_name='SIGSEGV')
    >>> decode_wait_status(3 << 8)                  # exit(3)
    DeathStatus(kind='exit', code=3, signal_name=None)

    A negative *status* (the ``subprocess`` convention for a signal death) is
    accepted too. Anything unparseable decodes to ``kind="unknown"``.
    """
    if status is None:
        return _UNKNOWN
    try:
        value = int(status)
    except (TypeError, ValueError):
        return _UNKNOWN
    if value < 0:
        return _signal_death(-value)
    try:
        if os.WIFSIGNALED(value):
            return _signal_death(os.WTERMSIG(value))
        if os.WIFEXITED(value):
            return DeathStatus("exit", os.WEXITSTATUS(value), None)
    except (OverflowError, ValueError):
        return _UNKNOWN
    return _UNKNOWN


def decode_returncode(code) -> DeathStatus:
    """Decode a ``subprocess.Popen`` return code / ``poll()`` result.

    ``None`` means "still running" and decodes to ``unknown`` (callers must
    not treat that as a death). A negative value is the ``subprocess``
    convention for "terminated by signal ``-code``".

    A positive value in the shell ``128 + N`` range is decoded as a **signal
    death**: a shell reports a foreground child killed by signal ``N`` with
    that exit status (which is exactly how the observed blanket ``137``
    masked a ``SIGSEGV``/``SIGKILL`` distinction). The heuristic is documented
    rather than hidden — a process that deliberately calls ``exit(137)`` is
    indistinguishable from one killed by ``SIGKILL`` at this layer, and
    reporting the signal is the more useful reading.
    """
    if code is None:
        return _UNKNOWN
    try:
        value = int(code)
    except (TypeError, ValueError):
        return _UNKNOWN
    if value < 0:
        return _signal_death(-value)
    if 128 < value <= 128 + MAX_SIGNAL_NUMBER and (value - 128) in _SIGNAL_NUMBERS:
        return _signal_death(value - 128)
    return DeathStatus("exit", value, None)


def peek_death_status(pid) -> DeathStatus | None:
    """Read *pid*'s raw wait status **without reaping** it.

    ``waitid(WNOWAIT)`` returns the raw status the ``Popen`` handle would
    otherwise reduce to a single integer: ``CLD_KILLED``/``CLD_DUMPED`` plus
    the signal number for a signal death, ``CLD_EXITED`` plus the exit status
    for a normal exit. Returns ``None`` while the process is still running, and
    when the pid is unusable (not a child, already reaped, no ``waitid``).
    """
    if pid is None:
        return None
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    if pid_int <= 0:
        return None
    try:
        info = os.waitid(
            os.P_PID, pid_int, os.WEXITED | os.WNOWAIT | os.WNOHANG
        )
    except (AttributeError, ChildProcessError, OSError, ValueError):
        return None
    if info is None:
        return None
    si_code = getattr(info, "si_code", None)
    si_status = getattr(info, "si_status", None)
    if si_status is None:
        return _UNKNOWN
    if si_code == os.CLD_EXITED:
        return DeathStatus("exit", int(si_status), None)
    if si_code in (os.CLD_KILLED, _CLD_DUMPED):
        return _signal_death(int(si_status))
    return _UNKNOWN


def _reap(proc) -> None:
    try:
        proc.poll()
    except Exception:  # pragma: no cover - a broken handle must not raise
        pass


def probe_death(proc) -> DeathStatus | None:
    """Return how *proc* ended, or ``None`` while it is still running.

    Prefers :func:`peek_death_status` (raw status, no reaping) so a wrapper's
    signal death is not collapsed into bash's ``128 + N`` exit code; falls back
    to ``Popen.poll()`` when no raw status is available (an adopted or mocked
    handle, or a child already reaped elsewhere).
    """
    if proc is None:
        return None
    status = peek_death_status(getattr(proc, "pid", None))
    if status is not None:
        _reap(proc)
        return status
    try:
        code = proc.poll()
    except Exception:
        return None
    if code is None:
        return None
    return decode_returncode(code)


# ---------------------------------------------------------------------------
# Host sampling (the death-window instrument)
# ---------------------------------------------------------------------------


def _server_cfg(config) -> dict:
    if isinstance(config, dict):
        section = config.get("server", {})
        if isinstance(section, dict):
            return section
    return {}


def read_process_rss_bytes(pid) -> int | None:
    """Resident set size of *pid* in bytes, or ``None`` when unreadable."""
    if pid is None:
        return None
    try:
        text = Path("/proc/%d/status" % int(pid)).read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError):
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
            return None
    return None


def read_cgroup_memory_events(path=None) -> dict[str, int]:
    """Parse ``cgroup v2 memory.events`` into ``{"oom": 0, "oom_kill": 3, ...}``.

    ``memory.events`` is the cgroup's own kill accounting — the record that
    distinguishes "the kernel OOM killer took these two processes" from "no
    cgroup event happened at all". Returns ``{}`` when the file is absent
    (cgroup v1 or a container without it).
    """
    target = Path(path) if path is not None else DEFAULT_CGROUP_MEMORY_EVENTS
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    events: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            events[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return events


@dataclass(frozen=True)
class CaptureSample:
    """One point-in-time observation in the death-window ring."""

    timestamp: float
    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
    gtt_used_bytes: int | None = None
    gtt_total_bytes: int | None = None
    processes: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    cgroup_memory_events: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "timestamp": round(float(self.timestamp), 3),
            "mem_total_bytes": self.mem_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "gtt_used_bytes": self.gtt_used_bytes,
            "gtt_total_bytes": self.gtt_total_bytes,
            "processes": {k: dict(v) for k, v in self.processes.items()},
            "cgroup_memory_events": dict(self.cgroup_memory_events),
        }


def _iter_targets(targets) -> list[tuple[str, object]]:
    if targets is None:
        return []
    if callable(targets):
        try:
            targets = targets()
        except Exception:
            return []
    try:
        return list(dict(targets).items())
    except (TypeError, ValueError):
        return []


def sample_host(targets=None, *, clock: Callable[[], float] = time.time) -> CaptureSample:
    """Read one :class:`CaptureSample` from ``/proc`` and the amdgpu sysfs.

    *targets* maps a label (``"llama_server"``, ``"tts_server"``, ``"proxy"``)
    to a pid, or is a zero-argument callable returning that mapping (so a
    restart between samples is picked up). Every read is best-effort.
    """
    from .host_pressure import read_memory_snapshot, read_oom_score_adj

    try:
        snapshot = read_memory_snapshot()
    except Exception:  # pragma: no cover - snapshot never raises by contract
        snapshot = None

    processes: dict[str, dict] = {}
    for label, pid in _iter_targets(targets):
        entry: dict[str, object] = {
            "pid": None,
            "rss_bytes": None,
            "oom_score_adj": None,
        }
        try:
            pid_int = None if pid is None else int(pid)
        except (TypeError, ValueError):
            pid_int = None
        if pid_int:
            entry["pid"] = pid_int
            entry["rss_bytes"] = read_process_rss_bytes(pid_int)
            try:
                entry["oom_score_adj"] = read_oom_score_adj(pid_int)
            except Exception:  # pragma: no cover - defensive
                entry["oom_score_adj"] = None
        processes[label] = entry

    return CaptureSample(
        timestamp=float(clock()),
        mem_total_bytes=getattr(snapshot, "mem_total_bytes", None),
        mem_available_bytes=getattr(snapshot, "mem_available_bytes", None),
        gtt_used_bytes=getattr(snapshot, "gtt_used_bytes", None),
        gtt_total_bytes=getattr(snapshot, "gtt_total_bytes", None),
        processes=processes,
        cgroup_memory_events=read_cgroup_memory_events(),
    )


def _summarise_samples(samples) -> dict:
    """Extremes across *samples* — the trend a reader needs, not 120 rows."""
    gtt_max = None
    mem_min = None
    rss_max: dict[str, int] = {}
    oom_last: dict[str, object] = {}
    events: dict[str, int] = {}
    for sample in samples:
        if sample.gtt_used_bytes is not None:
            gtt_max = (
                sample.gtt_used_bytes
                if gtt_max is None
                else max(gtt_max, sample.gtt_used_bytes)
            )
        if sample.mem_available_bytes is not None:
            mem_min = (
                sample.mem_available_bytes
                if mem_min is None
                else min(mem_min, sample.mem_available_bytes)
            )
        for label, entry in sample.processes.items():
            rss = entry.get("rss_bytes") if isinstance(entry, Mapping) else None
            if isinstance(rss, int):
                rss_max[label] = rss if label not in rss_max else max(rss_max[label], rss)
            if isinstance(entry, Mapping) and entry.get("oom_score_adj") is not None:
                oom_last[label] = entry.get("oom_score_adj")
        if sample.cgroup_memory_events:
            events = dict(sample.cgroup_memory_events)
    return {
        "gtt_used_max_bytes": gtt_max,
        "mem_available_min_bytes": mem_min,
        "process_rss_max_bytes": rss_max,
        "oom_score_adj": oom_last,
        "cgroup_memory_events": events,
    }


def _isoformat(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp), tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _gib(value) -> str:
    try:
        return "%.1fGiB" % (float(value) / (1024**3))
    except (TypeError, ValueError):
        return "unknown"


class DeathCapture:
    """Bounded ring of host samples plus the death-window record builder.

    *sampler* is called with the current *targets* and must return a
    :class:`CaptureSample` (the default is :func:`sample_host`). *clock* is
    injectable so a unit test can drive synthetic timestamps, which is how the
    ring and window behaviour is verified without waiting in real time.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
        sampler: Callable[..., CaptureSample] | None = None,
        max_record_samples: int = DEFAULT_MAX_RECORD_SAMPLES,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_record_samples = max(1, int(max_record_samples))
        self._clock = clock
        self._sampler = (
            sampler
            if sampler is not None
            else partial(sample_host, clock=clock)
        )
        self._samples: deque[CaptureSample] = deque(maxlen=self.capacity)
        self.deaths = 0
        self.last_death: dict | None = None

    def sample(self, targets=None) -> CaptureSample | None:
        """Take one sample and retain it. Never raises; returns ``None`` on failure."""
        try:
            snapshot = self._sampler(targets)
        except Exception:
            return None
        if snapshot is None:
            return None
        self._remember(snapshot)
        return snapshot

    def _remember(self, snapshot: CaptureSample) -> None:
        self._samples.append(snapshot)
        cutoff = float(snapshot.timestamp) - self.window_seconds
        while self._samples and float(self._samples[0].timestamp) < cutoff:
            self._samples.popleft()

    def samples(self) -> list[CaptureSample]:
        return list(self._samples)

    def window(self, *, now: float | None = None, seconds: float | None = None):
        """Samples inside the last *seconds* (default: the configured window)."""
        reference = float(self._clock() if now is None else now)
        span = self.window_seconds if seconds is None else float(seconds)
        cutoff = reference - span
        return [s for s in self._samples if float(s.timestamp) >= cutoff]

    def death_record(
        self,
        *,
        label: str,
        status: DeathStatus | None = None,
        now: float | None = None,
        extra: Mapping | None = None,
    ) -> dict:
        """Build the compact ``death_window`` record for a detected death."""
        detected = float(self._clock() if now is None else now)
        samples = self.window(now=detected)
        record = {
            "kind": "death_window",
            "label": label,
            "detected_at": _isoformat(detected),
            "detected_at_unix": round(detected, 3),
            "window_seconds": round(self.window_seconds, 3),
            "restart_reason": (
                status.restart_reason(label) if status is not None else None
            ),
            "death": status.as_dict() if status is not None else None,
            "summary": _summarise_samples(samples),
            "sample_count": len(samples),
            "samples": [s.as_dict() for s in samples[-self.max_record_samples :]],
        }
        if extra is not None:
            record["extra"] = dict(extra)
        self.deaths += 1
        self.last_death = record
        return record

    def format_record(self, record: Mapping, dump_path: str | None = None) -> str:
        """One compact log line for a death record."""
        death = record.get("death") or {}
        summary = record.get("summary") or {}
        parts = [
            "label=%s" % record.get("label"),
            "cause=%s" % (death.get("describe") or "unknown cause"),
            "restart_reason=%s" % (record.get("restart_reason") or "unknown"),
            "window=%ss" % record.get("window_seconds"),
            "samples=%s" % record.get("sample_count"),
            "gtt_max=%s" % _gib(summary.get("gtt_used_max_bytes")),
            "mem_available_min=%s" % _gib(summary.get("mem_available_min_bytes")),
        ]
        events = summary.get("cgroup_memory_events") or {}
        if events:
            parts.append(
                "cgroup_oom_kills=%s" % events.get("oom_kill", events.get("oom", 0))
            )
        if dump_path:
            parts.append("dump=%s" % dump_path)
        return " ".join(parts)

    def dump(
        self,
        record: Mapping,
        *,
        log_dir=None,
        logger=None,
        filename: str | None = None,
    ) -> str | None:
        """Persist *record* as JSON under *log_dir* and log one summary line.

        Returns the written path, or ``None`` when no directory was given or
        the write failed. Never raises: the death record must not turn a
        restart into a second failure.
        """
        path: str | None = None
        if log_dir:
            try:
                directory = Path(log_dir)
                directory.mkdir(parents=True, exist_ok=True)
                name = filename or "death_window_%s_%s.json" % (
                    record.get("label", "unknown"),
                    datetime.fromtimestamp(
                        float(record.get("detected_at_unix") or time.time()), tz=UTC
                    ).strftime("%Y%m%dT%H%M%SZ"),
                )
                target = directory / name
                target.write_text(
                    json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
                )
                path = str(target)
            except (OSError, TypeError, ValueError):
                path = None
        if logger is not None:
            try:
                logger.error("death_window %s", self.format_record(record, path))
            except Exception:  # pragma: no cover - logging must never raise
                pass
        return path


# ---------------------------------------------------------------------------
# Process-wide capture instance (armed from config)
# ---------------------------------------------------------------------------

_CAPTURE: DeathCapture | None = None


def get_death_capture(config=None) -> DeathCapture:
    """Return the process-wide :class:`DeathCapture`, creating it on first use."""
    global _CAPTURE
    if _CAPTURE is None:
        server_cfg = _server_cfg(config)
        try:
            capacity = int(
                server_cfg.get("death_capture_capacity", DEFAULT_CAPACITY)
                or DEFAULT_CAPACITY
            )
        except (TypeError, ValueError):
            capacity = DEFAULT_CAPACITY
        try:
            window = float(
                server_cfg.get("death_capture_window_seconds", DEFAULT_WINDOW_SECONDS)
                or DEFAULT_WINDOW_SECONDS
            )
        except (TypeError, ValueError):
            window = DEFAULT_WINDOW_SECONDS
        _CAPTURE = DeathCapture(capacity=capacity, window_seconds=window)
    return _CAPTURE


def reset_death_capture() -> None:
    """Drop the process-wide capture instance (used by tests and config reload)."""
    global _CAPTURE
    _CAPTURE = None


def _srv():
    """Lazy server-module access (mirrors ``backend_health._srv``)."""
    import proxy.server as _module

    return _module


def capture_targets(srv=None) -> dict[str, object]:
    """The processes the ring samples: proxy, llama-server and TTS server."""
    module = srv if srv is not None else _srv()
    targets: dict[str, object] = {"proxy": os.getpid()}
    for label, attr in (("llama_server", "llama_process"), ("tts_server", "tts_process")):
        try:
            targets[label] = getattr(getattr(module, attr, None), "pid", None)
        except Exception:  # pragma: no cover - defensive
            targets[label] = None
    return targets


def record_death(
    srv,
    label: str,
    status: DeathStatus | None,
    *,
    extra=None,
    state_attr: str | None = None,
) -> dict:
    """Sample once, build the ``death_window`` record, dump it, return it.

    The final sample matters most: the state *at* detection is the observation
    the ring may not have reached. The record is also kept in the service's
    recovery state (``backend_recovery_state`` / ``tts_recovery_state``) so
    ``/health`` exposes the last cause. Never raises.
    """
    capture = get_death_capture(getattr(srv, "config", None))
    try:
        capture.sample(capture_targets(srv))
    except Exception:  # pragma: no cover - sampling is best-effort
        pass
    record = capture.death_record(label=label, status=status, extra=extra)
    record["dump_path"] = capture.dump(
        record,
        log_dir=getattr(srv, "log_dir", None),
        logger=getattr(srv, "logger", None),
    )
    target_attr = state_attr or (
        "tts_recovery_state" if label.startswith("tts") else "backend_recovery_state"
    )
    try:
        getattr(srv, target_attr)["last_death"] = record
    except Exception:  # pragma: no cover - not every srv exposes the state dict
        pass
    return record


def _sampling_interval(config) -> float:
    try:
        interval = float(
            _server_cfg(config).get(
                "death_capture_interval_seconds", DEFAULT_INTERVAL_SECONDS
            )
            or DEFAULT_INTERVAL_SECONDS
        )
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_SECONDS
    return max(0.05, interval)


def death_capture_armed(config) -> bool:
    """``server.death_capture_enabled`` — off unless explicitly armed."""
    return bool(_server_cfg(config).get("death_capture_enabled", False))


async def death_capture_loop() -> None:
    """Sample the ring at the configured interval while armed.

    Exits immediately (like the TTS watchdog) when capture is not armed, so
    spawning the task unconditionally is safe.
    """
    srv = _srv()
    try:
        if not death_capture_armed(getattr(srv, "config", None)):
            srv.logger.info(
                "death capture: not armed (set server.death_capture_enabled=true "
                "to record a per-second death window)"
            )
            return
        capture = get_death_capture(getattr(srv, "config", None))
        interval = _sampling_interval(getattr(srv, "config", None))
        srv.logger.info(
            "death capture: armed (interval=%ss, window=%ss, capacity=%s, dir=%s)",
            interval,
            capture.window_seconds,
            capture.capacity,
            getattr(srv, "log_dir", None),
        )
        while True:
            try:
                await asyncio.sleep(interval)
                capture.sample(capture_targets(srv))
            except asyncio.CancelledError:
                srv.logger.info("death capture loop cancelled")
                return
            except Exception:
                srv.logger.exception("death capture loop error")
    except asyncio.CancelledError:  # pragma: no cover - startup race
        return
