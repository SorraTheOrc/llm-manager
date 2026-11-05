"""True death-signal decoding and death-window capture (LP-0MW3K7WD6003TJKZ).

The item exists because the proxy reported a blanket
``llama_server_exit_code_137`` for every backend death — 133 times across the
retained log rotations — while ``/var/log/kern.log`` recorded ``SIGSEGV`` /
general-protection faults for the *same* processes, and the TTS watchdog only
logged ``code=-9``. These tests pin:

* **AC1** — a synthetic ``SIGSEGV`` wait status decodes to ``SIGSEGV(11)/139``
  and a ``llama_server_signal_SIGSEGV`` restart reason, never ``137``;
* **AC2** — ``SIGKILL``/``SIGTERM``/normal exit stay distinct, so no death is
  reported as a blanket ``137``;
* **AC4** — the capture helper records the required fields (GPU GTT,
  ``MemAvailable``, per-process RSS/``oom_score_adj``, cgroup
  ``memory.events``) and dumps a bounded death window built from synthetic
  samples;
* the ring never grows past its capacity and a window never reaches further
  back than its configured span.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import proxy.death_capture as dc
import pytest


@pytest.fixture(autouse=True)
def _reset_capture():
    """Each test gets a fresh process-wide capture instance."""
    dc.reset_death_capture()
    yield
    dc.reset_death_capture()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_signal_status(signum, *, dumped=False):
    """Build the raw POSIX wait status of a process killed by *signum*.

    ``waitid``/``WTERMSIG`` read the low seven bits; ``WCOREDUMP`` is bit 7.
    """
    return int(signum) | (0x80 if dumped else 0)


class _Clock:
    def __init__(self, now: float = 1_000.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 1.0) -> None:
        self.now += float(seconds)


class _FixedSampler:
    """Sampler whose sample payload is controlled per call by the test."""

    def __init__(self, clock: _Clock, **values):
        self.clock = clock
        self.values = values
        self.calls = []

    def __call__(self, targets=None, **_kwargs):
        self.calls.append(targets)
        return dc.CaptureSample(timestamp=self.clock.now, **self.values)


class _GrowingSampler:
    """Sampler that makes every field move so a window summary is testable."""

    def __init__(self, clock: _Clock):
        self.clock = clock
        self.index = 0

    def __call__(self, targets=None, **_kwargs):
        self.index += 1
        step = self.index
        return dc.CaptureSample(
            timestamp=self.clock.now,
            mem_total_bytes=100_000,
            mem_available_bytes=100_000 - step,
            gtt_used_bytes=step * 100,
            gtt_total_bytes=1_000_000,
            processes={
                "llama_server": {
                    "pid": 4242,
                    "rss_bytes": step * 10,
                    "oom_score_adj": 800,
                }
            },
            cgroup_memory_events={"oom": 0, "oom_kill": step},
        )


# ---------------------------------------------------------------------------
# AC1/AC2 — raw POSIX wait-status decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signum,dumped,name,code",
    [
        (signal.SIGSEGV, True, "SIGSEGV", 139),
        (signal.SIGKILL, False, "SIGKILL", 137),
        (signal.SIGTERM, False, "SIGTERM", 143),
    ],
)
def test_raw_wait_status_decodes_the_true_signal(signum, dumped, name, code):
    """AC1/AC2: each signal decodes to its own name and 128+N code."""
    status = dc.decode_wait_status(_raw_signal_status(signum, dumped=dumped))

    assert status.kind == "signal"
    assert status.signal_name == name
    assert status.signal_number == int(signum)
    assert status.code == code
    assert status.describe() == f"{name}({int(signum)})/{code}"


def test_sigsegv_death_is_not_reported_as_exit_code_137():
    """AC1: the recorded cause is SIGSEGV/139, never the blanket ``137``."""
    status = dc.decode_wait_status(_raw_signal_status(signal.SIGSEGV, dumped=True))

    reason = status.restart_reason("llama_server")

    assert reason == "llama_server_signal_SIGSEGV"
    assert "137" not in reason
    assert "139" in status.describe()


def test_raw_normal_exit_status_stays_an_exit():
    """AC2: a normal exit decodes to its status, not to a signal."""
    status = dc.decode_wait_status(3 << 8)

    assert status == dc.DeathStatus("exit", 3, None)
    assert status.describe() == "exit code 3"
    assert status.restart_reason("tts_server") == "tts_server_exit_code_3"


def test_unparseable_status_is_unknown_not_a_guess():
    """An undecodable status is named ``unknown``, never a fabricated cause."""
    assert dc.decode_wait_status(None).kind == "unknown"
    assert dc.decode_wait_status("not-a-status").kind == "unknown"
    assert (
        dc.decode_wait_status(None).restart_reason("llama_server")
        == "llama_server_unknown_exit"
    )


@pytest.mark.parametrize(
    "returncode,kind,name,code",
    [
        (-signal.SIGSEGV, "signal", "SIGSEGV", 139),
        (137, "signal", "SIGKILL", 137),
        (139, "signal", "SIGSEGV", 139),
        (143, "signal", "SIGTERM", 143),
        (1, "exit", None, 1),
        (0, "exit", None, 0),
        (128, "exit", None, 128),
        (None, "unknown", None, None),
    ],
)
def test_popen_returncodes_decode_without_a_blanket_137(returncode, kind, name, code):
    """AC2: ``Popen`` return codes — including bash's ``128 + N`` — stay distinct."""
    status = dc.decode_returncode(returncode)

    assert status.kind == kind
    assert status.signal_name == name
    assert status.code == code


def test_peek_death_status_reads_the_raw_status_without_reaping():
    """The raw status is available *before* ``Popen`` collapses it.

    A real child is killed by ``SIGTERM``; ``waitid(WNOWAIT)`` must report the
    signal while leaving the child reapable, so the watchdog can name the cause
    and still use its ``Popen`` handle.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ]
    )
    try:
        deadline = time.time() + 15
        status = None
        while time.time() < deadline:
            status = dc.peek_death_status(proc.pid)
            if status is not None:
                break
            time.sleep(0.02)

        assert status is not None, "waitid never reported the child's death"
        assert status.kind == "signal"
        assert status.signal_name == "SIGTERM"
        # Peeking never reaped the child: the Popen handle still works.
        assert proc.wait(timeout=5) == -signal.SIGTERM
    finally:
        if proc.poll() is None:  # pragma: no cover - only on an assertion failure
            proc.kill()
            proc.wait(timeout=5)


def test_probe_death_is_none_while_the_process_is_alive():
    """A live process must never be classified as a death."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert dc.probe_death(proc) is None
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_probe_death_falls_back_to_the_returncode_for_an_unpeekable_handle():
    """An already-reaped / mocked handle still yields a decoded cause."""

    class _Handle:
        pid = None

        def poll(self):
            return 137

    status = dc.probe_death(_Handle())

    assert status is not None
    assert status.kind == "signal"
    assert status.signal_name == "SIGKILL"


# ---------------------------------------------------------------------------
# AC4 — host sampling and the death-window ring
# ---------------------------------------------------------------------------


def test_read_cgroup_memory_events_parses_event_counters(tmp_path):
    """cgroup ``memory.events`` is the cgroup's own kill accounting."""
    events_file = tmp_path / "memory.events"
    events_file.write_text(
        "low 0\nhigh 4\nmax 12\noom 1\noom_kill 2\noom_group_kill 0\n",
        encoding="utf-8",
    )

    events = dc.read_cgroup_memory_events(events_file)

    assert events["oom_kill"] == 2
    assert events["max"] == 12
    assert dc.read_cgroup_memory_events(tmp_path / "missing") == {}


def test_read_process_rss_bytes_reads_a_live_process():
    """Per-process RSS is recorded in bytes; an unusable pid yields ``None``."""
    rss = dc.read_process_rss_bytes(os.getpid())

    assert isinstance(rss, int) and rss > 0
    assert dc.read_process_rss_bytes(None) is None
    assert dc.read_process_rss_bytes(-1) is None


def test_sample_host_records_gtt_mem_rss_and_oom_score():
    """AC4: one sample carries every field the attribution work needs."""
    sample = dc.sample_host({"self": os.getpid(), "absent": None})

    assert isinstance(sample.timestamp, float)
    assert sample.mem_total_bytes is None or sample.mem_total_bytes > 0
    assert sample.processes["self"]["pid"] == os.getpid()
    assert isinstance(sample.processes["self"]["rss_bytes"], int)
    assert isinstance(sample.processes["self"]["oom_score_adj"], int)
    assert sample.processes["absent"] == {
        "pid": None,
        "rss_bytes": None,
        "oom_score_adj": None,
    }
    assert isinstance(sample.cgroup_memory_events, dict)


def test_ring_is_bounded_by_capacity_and_window():
    """AC4: the ring never grows past its cap and the window is time-bounded."""
    clock = _Clock()
    sampler = _FixedSampler(clock, mem_available_bytes=1)
    capture = dc.DeathCapture(
        capacity=5, window_seconds=3.0, clock=clock, sampler=sampler
    )

    for _ in range(12):
        clock.tick(0.5)
        capture.sample()

    assert len(capture.samples()) == 5  # capacity bound
    assert sampler.calls  # the injected sampler is what feeds the ring
    # Only the samples inside the requested span are returned.
    assert len(capture.window(seconds=1.0)) == 3


def test_death_record_summarises_the_window_and_bounds_its_samples():
    """AC4: a bounded window plus extremes, not an unbounded sample dump."""
    clock = _Clock()
    capture = dc.DeathCapture(
        capacity=200,
        window_seconds=60.0,
        clock=clock,
        sampler=_GrowingSampler(clock),
        max_record_samples=10,
    )
    for _ in range(100):
        clock.tick(1.0)
        capture.sample()

    status = dc.decode_wait_status(_raw_signal_status(signal.SIGSEGV, dumped=True))
    record = capture.death_record(
        label="llama_server", status=status, now=clock.now
    )

    window = capture.window(now=clock.now)
    assert record["kind"] == "death_window"
    assert record["label"] == "llama_server"
    assert record["restart_reason"] == "llama_server_signal_SIGSEGV"
    assert record["death"]["signal_name"] == "SIGSEGV"
    assert record["sample_count"] == len(window) < 100
    assert len(record["samples"]) == 10
    assert record["samples"][-1] == window[-1].as_dict()
    assert record["summary"]["gtt_used_max_bytes"] == max(
        s.gtt_used_bytes for s in window
    )
    assert record["summary"]["mem_available_min_bytes"] == min(
        s.mem_available_bytes for s in window
    )
    assert record["summary"]["process_rss_max_bytes"]["llama_server"] == max(
        s.processes["llama_server"]["rss_bytes"] for s in window
    )
    assert record["summary"]["oom_score_adj"]["llama_server"] == 800
    assert record["summary"]["cgroup_memory_events"]["oom_kill"] == 100
    assert capture.deaths == 1
    assert capture.last_death is record


def test_dump_writes_a_json_record_and_logs_one_line(tmp_path, caplog):
    """AC4/AC5: the death window is persisted and summarised in one log line."""
    clock = _Clock()
    capture = dc.DeathCapture(
        clock=clock, window_seconds=60.0, sampler=_GrowingSampler(clock)
    )
    clock.tick(1.0)
    capture.sample()
    record = capture.death_record(
        label="tts_server", status=dc.decode_returncode(-signal.SIGKILL)
    )

    logger = logging.getLogger("test.death_capture.dump")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        path = capture.dump(record, log_dir=tmp_path, logger=logger)

    assert path is not None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["kind"] == "death_window"
    assert payload["restart_reason"] == "tts_server_signal_SIGKILL"
    assert payload["summary"]["gtt_used_max_bytes"] == 100
    assert len(caplog.records) == 1
    assert "restart_reason=tts_server_signal_SIGKILL" in caplog.text
    assert "dump=" + path in caplog.text


def test_dump_is_silent_and_successful_without_a_log_dir(caplog):
    """No ``log_dir`` means no file: the record is only logged, never raised."""
    clock = _Clock()
    capture = dc.DeathCapture(clock=clock, sampler=_FixedSampler(clock))
    record = capture.death_record(
        label="llama_server", status=dc.decode_returncode(137)
    )

    logger = logging.getLogger("test.death_capture.nodir")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        path = capture.dump(record, log_dir=None, logger=logger)

    assert path is None
    assert "cause=SIGKILL(9)/137" in caplog.text


def test_record_death_persists_state_and_recovers_the_ring(tmp_path):
    """A death record is exposed on the service's recovery state."""

    class _Srv:
        config = {"server": {"death_capture_capacity": 5}}
        log_dir = tmp_path
        logger = logging.getLogger("test.death_capture.record")
        backend_recovery_state = {"last_failure": None}
        tts_recovery_state = {"last_failure": None}
        llama_process = None
        tts_process = None

    srv = _Srv()
    status = dc.decode_wait_status(_raw_signal_status(signal.SIGSEGV, dumped=True))

    record = dc.record_death(
        srv,
        "llama_server",
        status,
        extra={"model": "qwen"},
        state_attr="backend_recovery_state",
    )

    assert srv.backend_recovery_state["last_death"] is record
    assert record["restart_reason"] == "llama_server_signal_SIGSEGV"
    assert record["extra"] == {"model": "qwen"}
    assert record["sample_count"] >= 1  # the at-detection sample
    assert record["dump_path"] and Path(record["dump_path"]).exists()


def test_capture_is_not_armed_by_default():
    """AC4: capture is armable, not always-on."""
    assert dc.death_capture_armed({}) is False
    assert dc.death_capture_armed({"server": {}}) is False
    assert dc.death_capture_armed({"server": {"death_capture_enabled": True}}) is True


@pytest.mark.asyncio
async def test_capture_loop_exits_immediately_when_not_armed(monkeypatch):
    """The sampling task self-disables, so spawning it unconditionally is safe."""
    logged: list[str] = []

    class _Srv:
        config = {"server": {}}

        class logger:  # noqa: N801 - mirrors the server module's attribute
            @staticmethod
            def info(message, *args):
                logged.append(message % args if args else message)

    monkeypatch.setattr(dc, "_srv", lambda: _Srv())

    await dc.death_capture_loop()  # returns instead of looping

    assert any("not armed" in line for line in logged)


@pytest.mark.asyncio
async def test_capture_loop_samples_at_the_configured_interval(monkeypatch):
    """AC4: while armed, the ring is fed at the configured interval."""
    import asyncio

    class _Srv:
        config = {
            "server": {
                "death_capture_enabled": True,
                "death_capture_interval_seconds": 0.05,
            }
        }
        log_dir = None
        llama_process = None
        tts_process = None

        class logger:  # noqa: N801
            @staticmethod
            def info(*_args, **_kwargs):
                pass

            @staticmethod
            def exception(*_args, **_kwargs):  # pragma: no cover - failure path
                pass

    clock = _Clock()
    capture = dc.DeathCapture(clock=clock, sampler=_FixedSampler(clock))
    monkeypatch.setattr(dc, "_srv", lambda: _Srv())
    monkeypatch.setattr(dc, "get_death_capture", lambda *_a, **_k: capture)

    task = asyncio.create_task(dc.death_capture_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(capture.samples()) >= 1
