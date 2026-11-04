"""Host memory pressure: OOM victim ranking and pre-OOM observability.

Covers ``proxy.host_pressure`` (LP-0MVZ810NG008A3EW AC1 + AC3). The incident
numbers are the fixtures: on 2026-11-02T12:03:08Z the kernel hit a *global* OOM
on this host with

* ``MemTotal 130,494,996 kB`` and ``MemAvailable 245,576 kB`` (0.2 %),
* ``gpu_active 101,385,112 kB`` GPU-backed (amdgpu GTT) memory with
  ``gpu_reclaim 0``,
* llama-server pid 3702214 at 17,109,528 kB RSS and the proxy pid 3698721 at
  3,723,764 kB RSS, **both** at ``oom_score_adj`` 0.

The selection tests reconstruct that event with the kernel's own
``oom_badness()`` heuristic instead of provoking a real global OOM (which would
risk killing the live proxy). The write tests use a real child process for the
half of the protection that is always permitted (raising a score); the
privileged half (lowering below 0) is pinned through a simulated refusal, since
a test suite cannot acquire ``CAP_SYS_RESOURCE``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import proxy.host_pressure as hp
import pytest

# Observed at 2026-11-02T12:03:08Z (/var/log/kern.log).
INCIDENT_MEM_TOTAL = 130_494_996 * 1024
INCIDENT_MEM_AVAILABLE = 245_576 * 1024
INCIDENT_GTT_USED = 68_482 * 1024 * 1024  # 68,482 MiB measured 12:10Z
INCIDENT_GTT_TOTAL = 131_072 * 1024 * 1024  # 128 GiB pool
INCIDENT_PROXY_RSS = 3_723_764 * 1024
INCIDENT_BACKEND_RSS = 17_109_528 * 1024


# ---------------------------------------------------------------------------
# Meminfo / snapshot parsing
# ---------------------------------------------------------------------------


def test_read_meminfo_converts_kb_and_skips_counts(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       130494996 kB\n"
        "MemFree:          100000 kB\n"
        "MemAvailable:     245576 kB\n"
        "HugePages_Total:       0\n"
        "Hugepagesize:       2048 kB\n"
        "Garbage: not-a-number kB\n",
        encoding="utf-8",
    )

    values = hp.read_meminfo(meminfo)

    assert values["MemTotal"] == 130_494_996 * 1024
    assert values["MemAvailable"] == 245_576 * 1024
    assert values["Hugepagesize"] == 2048 * 1024
    # A page count is not a byte quantity.
    assert "HugePages_Total" not in values
    assert "Garbage" not in values


def test_read_meminfo_missing_file_is_empty(tmp_path):
    assert hp.read_meminfo(tmp_path / "nope") == {}


def test_snapshot_reads_gtt_and_ratios(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal: {INCIDENT_MEM_TOTAL // 1024} kB\n"
        f"MemAvailable: {INCIDENT_MEM_AVAILABLE // 1024} kB\n",
        encoding="utf-8",
    )
    device = tmp_path / "card0"
    device.mkdir()
    (device / "mem_info_gtt_used").write_text(str(INCIDENT_GTT_USED), encoding="utf-8")
    (device / "mem_info_gtt_total").write_text(str(INCIDENT_GTT_TOTAL), encoding="utf-8")

    snapshot = hp.read_memory_snapshot(meminfo, device)

    assert snapshot.mem_total_bytes == INCIDENT_MEM_TOTAL
    assert snapshot.mem_available_bytes == INCIDENT_MEM_AVAILABLE
    assert snapshot.gtt_total_bytes == INCIDENT_GTT_TOTAL
    assert snapshot.available_fraction == pytest.approx(0.00188, abs=1e-5)
    # GTT is compared with physical RAM: 68,482 MiB of 124 GiB = 53.7 %.
    assert snapshot.gtt_host_fraction == pytest.approx(0.5374, abs=1e-3)


def test_snapshot_missing_sysfs_reports_none_not_zero(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1024 kB\nMemAvailable: 512 kB\n", encoding="utf-8")

    snapshot = hp.read_memory_snapshot(meminfo, tmp_path / "absent")

    assert snapshot.gtt_used_bytes is None
    assert snapshot.gtt_host_fraction is None


# ---------------------------------------------------------------------------
# Pressure assessment (AC3: warn before the kernel watermark)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "available_fraction,expected_level",
    [
        (0.50, "ok"),
        (0.12, "warning"),  # exactly at the warning threshold
        (0.08, "warning"),
        (0.06, "critical"),
        (0.002, "critical"),  # the 2026-11-02 value
    ],
)
def test_available_memory_thresholds(available_fraction, expected_level):
    snapshot = hp.MemorySnapshot(
        mem_total_bytes=INCIDENT_MEM_TOTAL,
        mem_available_bytes=int(INCIDENT_MEM_TOTAL * available_fraction),
    )
    assert hp.evaluate_pressure(snapshot).level == expected_level


@pytest.mark.parametrize(
    "gtt_fraction,expected_level",
    [
        (0.30, "ok"),
        (0.50, "warning"),
        (0.5374, "warning"),  # the 2026-11-02 GTT/RAM ratio
        (0.66, "critical"),
    ],
)
def test_gtt_vs_ram_thresholds(gtt_fraction, expected_level):
    snapshot = hp.MemorySnapshot(
        mem_total_bytes=INCIDENT_MEM_TOTAL,
        mem_available_bytes=INCIDENT_MEM_TOTAL // 2,
        gtt_used_bytes=int(INCIDENT_MEM_TOTAL * gtt_fraction),
    )
    assert hp.evaluate_pressure(snapshot).level == expected_level


def test_incident_snapshot_is_critical_with_both_reasons():
    """The 2026-11-02 state must be alarming on both series, with reasons named."""
    snapshot = hp.MemorySnapshot(
        mem_total_bytes=INCIDENT_MEM_TOTAL,
        mem_available_bytes=INCIDENT_MEM_AVAILABLE,
        gtt_used_bytes=INCIDENT_GTT_USED,
        gtt_total_bytes=INCIDENT_GTT_TOTAL,
    )

    verdict = hp.evaluate_pressure(snapshot)

    assert verdict.level == "critical"
    joined = " ".join(verdict.reasons)
    assert "MemAvailable" in joined
    assert "GPU GTT" in joined


def test_unknown_when_nothing_is_measurable():
    verdict = hp.evaluate_pressure(hp.MemorySnapshot())
    assert verdict.level == "unknown"
    assert not verdict.is_alarming


def test_log_pressure_warns_at_warning_and_never_raises():
    logger = logging.getLogger("test-host-pressure")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        snapshot = hp.MemorySnapshot(
            mem_total_bytes=INCIDENT_MEM_TOTAL,
            mem_available_bytes=int(INCIDENT_MEM_TOTAL * 0.002),
        )
        verdict = hp.log_pressure(logger, snapshot, context="unit-test")
    finally:
        logger.removeHandler(handler)

    assert verdict.level == "critical"
    assert any(r.levelno == logging.WARNING for r in records)
    assert any("unit-test" in r.getMessage() for r in records)


def test_log_pressure_swallows_a_broken_logger():
    class _Broken:
        def warning(self, *a, **kw):
            raise RuntimeError("no logging for you")

        def info(self, *a, **kw):
            raise RuntimeError("no logging for you")

        def debug(self, *a, **kw):
            raise RuntimeError("no logging for you")

    verdict = hp.log_pressure(_Broken(), hp.MemorySnapshot(mem_total_bytes=1, mem_available_bytes=1))
    assert verdict.level == "ok"


# ---------------------------------------------------------------------------
# OOM victim ranking (AC1)
# ---------------------------------------------------------------------------


def _incident_candidates(proxy_adj: int, backend_adj: int) -> list[hp.OomCandidate]:
    return [
        hp.OomCandidate(3698721, "proxy", INCIDENT_PROXY_RSS, proxy_adj),
        hp.OomCandidate(3702214, "llama-server", INCIDENT_BACKEND_RSS, backend_adj),
    ]


def test_badness_scales_rss_and_oom_score_adj():
    candidate = hp.OomCandidate(1, "x", INCIDENT_MEM_TOTAL // 10, 0)
    assert hp.oom_badness(candidate, INCIDENT_MEM_TOTAL) == pytest.approx(100.0)
    # +500 multiplies the score by 1.5, -500 halves it.
    assert hp.oom_badness(
        hp.OomCandidate(1, "x", INCIDENT_MEM_TOTAL // 10, 500), INCIDENT_MEM_TOTAL
    ) == pytest.approx(150.0)
    assert hp.oom_badness(
        hp.OomCandidate(1, "x", INCIDENT_MEM_TOTAL // 10, -500), INCIDENT_MEM_TOTAL
    ) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        hp.oom_badness(candidate, 0)


def test_opt_out_score_is_never_a_victim():
    candidate = hp.OomCandidate(1, "x", INCIDENT_MEM_TOTAL, hp.OOM_SCORE_ADJ_MIN)
    assert candidate.killable is False
    assert hp.oom_badness(candidate, INCIDENT_MEM_TOTAL) is None
    assert hp.select_oom_victim([candidate], INCIDENT_MEM_TOTAL) is None


def test_unprotected_incident_kills_the_backend_first_then_the_proxy():
    """Pre-fix state: both at score 0, so a global OOM has both on its list."""
    ranked = hp.rank_oom_victims(_incident_candidates(0, 0), INCIDENT_MEM_TOTAL)

    assert [c.name for c, _ in ranked] == ["llama-server", "proxy"]
    assert ranked[0][1] > ranked[1][1]


def test_protected_incident_selects_only_the_backend_first():
    """AC1: with the shipped defaults the proxy's badness collapses."""
    ranked = hp.rank_oom_victims(
        _incident_candidates(
            hp.PROXY_OOM_SCORE_ADJ, hp.LLAMA_SERVER_OOM_SCORE_ADJ
        ),
        INCIDENT_MEM_TOTAL,
    )

    victim = ranked[0][0]
    assert victim.name == "llama-server", "the front door would still be the victim"
    proxy_score = dict((c.name, s) for c, s in ranked)["proxy"]
    backend_score = dict((c.name, s) for c, s in ranked)["llama-server"]
    assert backend_score > 5 * proxy_score, (
        "the backend must dominate the ranking, not merely edge ahead: "
        f"backend={backend_score} proxy={proxy_score}"
    )


def test_unprivileged_deployment_still_ranks_backend_above_proxy():
    """The permitted half of the protection must be enough on its own.

    A user-level deployment cannot lower the proxy's score, so the proxy stays
    at 0 and only the backend is raised. That alone must keep the proxy behind
    the backend in the kill order — otherwise the protection is useless exactly
    where it matters most.
    """
    ranked = hp.rank_oom_victims(
        _incident_candidates(0, hp.LLAMA_SERVER_OOM_SCORE_ADJ), INCIDENT_MEM_TOTAL
    )
    assert [c.name for c, _ in ranked] == ["llama-server", "proxy"]


# ---------------------------------------------------------------------------
# Applying the adjustment
# ---------------------------------------------------------------------------


def _spawn_sleep() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "60"])


def test_apply_oom_score_adj_raises_a_real_process_and_reads_it_back():
    """Raising a score is always permitted; the read-back is the applied value."""
    proc = _spawn_sleep()
    try:
        applied, error = hp.apply_oom_score_adj(proc.pid, hp.LLAMA_SERVER_OOM_SCORE_ADJ)
        assert error is None
        assert applied == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert hp.read_oom_score_adj(proc.pid) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_apply_oom_score_adj_clamps_to_kernel_bounds():
    proc = _spawn_sleep()
    try:
        applied, error = hp.apply_oom_score_adj(proc.pid, 99_999)
        assert error is None
        assert applied == hp.OOM_SCORE_ADJ_MAX
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_apply_oom_score_adj_reports_a_dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait(timeout=10)
    applied, error = hp.apply_oom_score_adj(proc.pid, 100)
    assert error is not None
    assert applied is None


def test_protect_backend_process_returns_the_applied_score():
    proc = _spawn_sleep()
    try:
        applied = hp.protect_backend_process(proc, {"server": {}}, None)
        assert applied == hp.LLAMA_SERVER_OOM_SCORE_ADJ
        assert hp.read_oom_score_adj(proc.pid) == hp.LLAMA_SERVER_OOM_SCORE_ADJ
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_protect_backend_process_honours_config_and_missing_pid():
    proc = _spawn_sleep()
    try:
        applied = hp.protect_backend_process(
            proc, {"server": {"llama_server_oom_score_adj": 300}}, None
        )
        assert applied == 300
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    class _NoPid:
        pid = 0

    assert hp.protect_backend_process(_NoPid(), {"server": {}}, None) is None


def test_protect_proxy_process_applies_and_reports(caplog):
    """A permitted value is applied to *this* process and reported at INFO."""
    original = hp.read_oom_score_adj(os.getpid())
    logger = logging.getLogger("test-protect-proxy")
    try:
        with caplog.at_level(logging.INFO, logger="test-protect-proxy"):
            applied = hp.protect_proxy_process({"server": {"oom_score_adj": 0}}, logger)
        assert applied == 0
        assert hp.read_oom_score_adj(os.getpid()) == 0
    finally:
        if original is not None:
            hp.apply_oom_score_adj(os.getpid(), original)


def test_protect_proxy_process_explains_a_refused_negative_score(caplog, monkeypatch):
    """AC1: the unprivileged refusal is loud, actionable and non-fatal."""
    monkeypatch.setattr(
        hp, "apply_oom_score_adj", lambda pid, value: (0, "PermissionError: Operation not permitted")
    )
    logger = logging.getLogger("test-protect-proxy-refused")
    with caplog.at_level(logging.WARNING, logger="test-protect-proxy-refused"):
        applied = hp.protect_proxy_process({"server": {"oom_score_adj": -900}}, logger)

    assert applied == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "CAP_SYS_RESOURCE" in messages
    assert "OOMScoreAdjust=-900" in messages


# ---------------------------------------------------------------------------
# Recovery measurement (AC2)
# ---------------------------------------------------------------------------


def test_measure_startup_gap_uses_the_newest_archived_log(tmp_path):
    archived = tmp_path / "proxy.log.2026-11-02_120245"
    archived.write_text("last pre-OOM line\n", encoding="utf-8")
    last_write = 1_793_000_000.0
    os.utime(archived, (last_write, last_write))

    measured = hp.measure_startup_gap(tmp_path, now=last_write + 650)

    assert measured is not None
    assert measured["gap_seconds"] == pytest.approx(650.0, abs=1.0)
    assert measured["source"] == archived.name
    assert measured["last_write"].startswith("2026-")


def test_measure_startup_gap_falls_back_to_the_live_log(tmp_path):
    live = tmp_path / "proxy.log"
    live.write_text("x\n", encoding="utf-8")
    os.utime(live, (1000.0, 1000.0))

    measured = hp.measure_startup_gap(tmp_path, now=1060.0)

    assert measured is not None
    assert measured["source"] == "proxy.log"
    assert measured["gap_seconds"] == pytest.approx(60.0)


def test_measure_startup_gap_none_without_logs(tmp_path):
    assert hp.measure_startup_gap(tmp_path) is None
    assert hp.measure_startup_gap(None) is None


def test_log_startup_gap_warns_on_a_long_outage(tmp_path, caplog):
    archived = tmp_path / "proxy.log.2026-11-02_120245"
    archived.write_text("x\n", encoding="utf-8")
    os.utime(archived, (1000.0, 1000.0))
    logger = logging.getLogger("test-startup-gap")

    with caplog.at_level(logging.WARNING, logger="test-startup-gap"):
        measured = hp.log_startup_gap(logger, tmp_path, now=1000.0 + 650)

    assert measured is not None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any("Restart=always" in r.getMessage() for r in caplog.records)


def test_log_startup_gap_is_info_for_a_quick_restart(tmp_path, caplog):
    archived = tmp_path / "proxy.log.2026-11-02_120245"
    archived.write_text("x\n", encoding="utf-8")
    os.utime(archived, (1000.0, 1000.0))
    logger = logging.getLogger("test-startup-gap-fast")

    with caplog.at_level(logging.INFO, logger="test-startup-gap-fast"):
        hp.log_startup_gap(logger, tmp_path, now=1005.0)

    assert [r.levelno for r in caplog.records] == [logging.INFO]


# ---------------------------------------------------------------------------
# /metrics publication (AC3)
# ---------------------------------------------------------------------------


class _RecordingGauge:
    def __init__(self):
        self.values: list[float] = []

    def set(self, value):
        self.values.append(value)


def _fake_gauges(monkeypatch):
    import proxy.metrics as metrics

    gauges = {
        "llama_host_mem_available_ratio": _RecordingGauge(),
        "llama_host_gtt_used_ratio_of_ram": _RecordingGauge(),
        "llama_host_memory_sample_timestamp_seconds": _RecordingGauge(),
        "llama_oom_score_adj": _RecordingGauge(),
    }
    for name, gauge in gauges.items():
        monkeypatch.setattr(metrics, name, gauge)
    monkeypatch.setattr(metrics, "_enabled", True)
    return metrics, gauges


def test_record_host_pressure_publishes_ratios_and_heartbeat(monkeypatch):
    metrics, gauges = _fake_gauges(monkeypatch)
    snapshot = hp.MemorySnapshot(
        mem_total_bytes=INCIDENT_MEM_TOTAL,
        mem_available_bytes=INCIDENT_MEM_AVAILABLE,
        gtt_used_bytes=INCIDENT_GTT_USED,
    )

    metrics.record_host_pressure(snapshot, 1_700_000_000.0, oom_score_adj=-900)

    assert gauges["llama_host_mem_available_ratio"].values == [
        pytest.approx(0.00188, abs=1e-5)
    ]
    assert gauges["llama_host_gtt_used_ratio_of_ram"].values == [
        pytest.approx(0.5374, abs=1e-3)
    ]
    assert gauges["llama_host_memory_sample_timestamp_seconds"].values == [1_700_000_000.0]
    assert gauges["llama_oom_score_adj"].values == [-900]


def test_record_host_pressure_does_not_publish_an_unmeasurable_host(monkeypatch):
    """A container without amdgpu sysfs must not look like an exhausted host."""
    metrics, gauges = _fake_gauges(monkeypatch)

    metrics.record_host_pressure(hp.MemorySnapshot(), 1_700_000_000.0)

    assert gauges["llama_host_mem_available_ratio"].values == []
    assert gauges["llama_host_gtt_used_ratio_of_ram"].values == []
    assert gauges["llama_host_memory_sample_timestamp_seconds"].values == [], (
        "the heartbeat must not advance when nothing was measured"
    )
