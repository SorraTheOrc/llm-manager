
# <!-- REFACTOR-LP-0MVZA6FV20077E6I
# smell: modernization
# severity: medium
# description: Use format specifiers instead of percent format
# -->
"""Host memory-pressure protection and pre-OOM observability.

Two incidents motivated this module (LP-0MVZ810NG008A3EW, 2026-11-02T12:03:08Z):

* the kernel hit a **global** OOM: ~96.7 GiB of the 124 GiB host was held as
  GPU-backed memory (``amdgpu`` GTT, which the kernel reports as
  ``gpu_active``) while ``gpu_reclaim`` was 0. Anonymous memory sat on top,
  Node 0 fell below its ``min`` watermark, and the OOM killer SIGKILLed both
  the proxy (pid 3698721) and its llama-server backend (pid 3702214) — 30 kill
  lines, one timestamp, ~10.5 minutes of connection-refused before an operator
  restarted the proxy by hand;
* nothing recorded the pressure *before* the kill: ``ps`` RSS accounting cannot
  see GTT, ``gpu_busy_percent`` is useless on this APU, and the only evidence
  left was a kernel log line.

The module therefore has two halves:

OOM victim protection (:func:`protect_proxy_process`,
:func:`protect_backend_process`)
    Give the llama-server backend a high ``oom_score_adj`` (raising is always
    permitted) and try to give the proxy a negative one. The kernel's
    ``oom_badness`` scaling then makes the *backend* the first victim instead of
    the front door. Lowering ``oom_score_adj`` below 0 needs
    ``CAP_SYS_RESOURCE``; when the proxy runs unprivileged the attempt fails
    safely and only the backend half is applied — the proxy then still outlives
    the backend because the backend's score dominates. :func:`oom_badness` and
    :func:`select_oom_victim` implement the kernel's selection heuristic so the
    protection is verifiable without provoking a real global OOM (which would
    risk killing the very proxy under test).

Pre-OOM observability (:func:`read_memory_snapshot`, :func:`evaluate_pressure`)
    ``MemAvailable`` from ``/proc/meminfo`` plus GTT use/size from the
    ``amdgpu`` sysfs attributes, evaluated against thresholds that trip *before*
    the kernel watermark. The proxy logs the assessment at startup and exports
    it on ``/metrics`` (see ``proxy.metrics.record_host_pressure``), so a load
    event that is about to tip the host over is visible as a warning instead of
    only as a post-mortem kernel line.

Everything here is best-effort and dependency-light (stdlib only): a missing
``/proc`` entry, an unreadable sysfs attribute or an unprivileged write must
never break proxy startup.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (overridable through the ``server:`` config block)
# ---------------------------------------------------------------------------

#: Default proxy ``oom_score_adj``. Negative values require ``CAP_SYS_RESOURCE``;
#: the systemd unit sets ``OOMScoreAdjust=-900`` for privileged deployments and
#: the runtime attempt degrades to the current value when it is refused.
PROXY_OOM_SCORE_ADJ = -900

#: Default llama-server ``oom_score_adj``. Raising a child's score is always
#: permitted, so this half of the protection works unprivileged.
LLAMA_SERVER_OOM_SCORE_ADJ = 800

#: ``MemAvailable`` fractions at which the host is considered close to the
#: kernel watermark (the 2026-11-02 event ran at ``245576 kB`` available of
#: ``130494996 kB`` total — 0.2 %, far below both).
MEM_AVAILABLE_WARN_FRACTION = 0.12
MEM_AVAILABLE_CRITICAL_FRACTION = 0.06

#: GTT use as a fraction of *physical RAM* (not of the GTT pool: the pool is
#: 128 GiB on a 124 GiB host, so a pool-relative threshold can never fire early
#: enough). The incident sat at ``68482 MiB / 130495 MiB`` = 0.52 with
#: ``gpu_reclaim`` 0, i.e. over the warning threshold below.
GTT_HOST_WARN_FRACTION = 0.50
GTT_HOST_CRITICAL_FRACTION = 0.65

DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")
DEFAULT_GPU_SYSFS = Path("/sys/class/drm/card0/device")

#: Kernel ``oom_score_adj`` bounds.
OOM_SCORE_ADJ_MIN = -1000
OOM_SCORE_ADJ_MAX = 1000


# ---------------------------------------------------------------------------
# Memory snapshot (MemAvailable + GTT)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemorySnapshot:
    """One point-in-time view of host memory pressure.

    Any field may be ``None``: a missing ``/proc`` entry or sysfs attribute
    means "not measurable", never zero (a zero ``MemAvailable`` would trip an
    alert, a zero GTT would hide one).
    """

    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
    gtt_used_bytes: int | None = None
    gtt_total_bytes: int | None = None

    @property
    def available_fraction(self) -> float | None:
        """``MemAvailable / MemTotal``, or ``None`` when either is unknown."""
        if not self.mem_total_bytes or self.mem_available_bytes is None:
            return None
        return self.mem_available_bytes / self.mem_total_bytes

    @property
    def gtt_host_fraction(self) -> float | None:
        """GTT in use as a fraction of physical RAM (``None`` when unknown)."""
        if not self.mem_total_bytes or self.gtt_used_bytes is None:
            return None
        return self.gtt_used_bytes / self.mem_total_bytes


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_meminfo(meminfo_path=None) -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a ``{key: bytes}`` dict.

    Values in ``/proc/meminfo`` are in kB; this converts to bytes. Unparseable
    lines (and the ``HugePages_*`` counters, which are counts not kB) are
    skipped.
    """
    path = Path(meminfo_path) if meminfo_path is not None else DEFAULT_MEMINFO_PATH
    values: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        key = key.strip()
        # ``HugePages_*`` are page counts, not byte quantities (their unitless
        # value would otherwise be read as bytes).
        if key.startswith("HugePages_"):
            continue
        fields = rest.split()
        if not fields:
            continue
        try:
            number = int(fields[0])
        except ValueError:
            continue
        unit = fields[1] if len(fields) > 1 else ""
        if unit == "kB":
            number *= 1024
        elif unit:
            # Counts (e.g. ``HugePages_Total: 0``) are not byte quantities.
            continue
        values[key] = number
    return values


def read_memory_snapshot(meminfo_path=None, gpu_sysfs=None) -> MemorySnapshot:
    """Read one :class:`MemorySnapshot` from ``/proc/meminfo`` + amdgpu sysfs."""
    meminfo = read_meminfo(meminfo_path)
    device = Path(gpu_sysfs) if gpu_sysfs is not None else DEFAULT_GPU_SYSFS
    return MemorySnapshot(
        mem_total_bytes=meminfo.get("MemTotal"),
        mem_available_bytes=meminfo.get("MemAvailable"),
        gtt_used_bytes=_read_int(device / "mem_info_gtt_used"),
        gtt_total_bytes=_read_int(device / "mem_info_gtt_total"),
    )


@dataclass(frozen=True)
class PressureAssessment:
    """Verdict for one :class:`MemorySnapshot`."""

    level: str  # "ok" | "warning" | "critical" | "unknown"
    reasons: tuple[str, ...] = ()

    @property
    def is_alarming(self) -> bool:
        return self.level in ("warning", "critical")


def evaluate_pressure(
    snapshot: MemorySnapshot,
    *,
    available_warn_fraction: float = MEM_AVAILABLE_WARN_FRACTION,
    available_critical_fraction: float = MEM_AVAILABLE_CRITICAL_FRACTION,
    gtt_warn_fraction: float = GTT_HOST_WARN_FRACTION,
    gtt_critical_fraction: float = GTT_HOST_CRITICAL_FRACTION,
) -> PressureAssessment:
    """Classify *snapshot* against the pre-OOM thresholds.

    ``MemAvailable`` is the authoritative headroom signal; GTT-vs-RAM is the
    leading indicator specific to this APU (GPU-backed memory is invisible to
    RSS accounting, so ordinary process-level monitoring misses it). Both are
    compared against *physical RAM*: the GTT pool is larger than RAM on this
    host, so a pool-relative threshold lags the real danger by tens of GiB.
    """
    critical: list[str] = []
    warning: list[str] = []

    available_fraction = snapshot.available_fraction
    if available_fraction is not None:
        if available_fraction <= available_critical_fraction:
            critical.append(
                "MemAvailable %.1f%% of MemTotal (critical threshold %.0f%%)"
                % (available_fraction * 100, available_critical_fraction * 100)
            )
        elif available_fraction <= available_warn_fraction:
            warning.append(
                "MemAvailable %.1f%% of MemTotal (warning threshold %.0f%%)"
                % (available_fraction * 100, available_warn_fraction * 100)
            )

    gtt_fraction = snapshot.gtt_host_fraction
    if gtt_fraction is not None:
        if gtt_fraction >= gtt_critical_fraction:
            critical.append(
                "GPU GTT %.1f%% of physical RAM (critical threshold %.0f%%)"
                % (gtt_fraction * 100, gtt_critical_fraction * 100)
            )
        elif gtt_fraction >= gtt_warn_fraction:
            warning.append(
                "GPU GTT %.1f%% of physical RAM (warning threshold %.0f%%)"
                % (gtt_fraction * 100, gtt_warn_fraction * 100)
            )

    if critical:
        return PressureAssessment("critical", tuple(critical + warning))
    if warning:
        return PressureAssessment("warning", tuple(warning))
    if available_fraction is None and gtt_fraction is None:
        return PressureAssessment("unknown", ("no memory series readable",))
    return PressureAssessment("ok")


def describe_snapshot(snapshot: MemorySnapshot) -> str:
    """Compact log representation of the measured series."""

    def _gib(value: int | None) -> str:
        return "unknown" if value is None else "%.1f GiB" % (value / (1024**3))

    return (
        "MemAvailable %s of %s, GPU GTT %s of %s"
        % (
            _gib(snapshot.mem_available_bytes),
            _gib(snapshot.mem_total_bytes),
            _gib(snapshot.gtt_used_bytes),
            _gib(snapshot.gtt_total_bytes),
        )
    )


def log_pressure(
    logger,
    snapshot: MemorySnapshot,
    *,
    context: str = "startup",
    assessment: PressureAssessment | None = None,
) -> PressureAssessment:
    """Log the pressure assessment, warning before the kernel watermark.

    A ``warning``/``critical`` verdict is logged at WARNING (it is the only
    pre-OOM signal that exists at the moment a model load is about to tip the
    host over); ``ok`` is logged at INFO and ``unknown`` at DEBUG so a container
    without sysfs does not spam the log. Never raises.
    """
    verdict = assessment if assessment is not None else evaluate_pressure(snapshot)
    message = "Host memory pressure [%s]: %s — %s" % (
        context,
        verdict.level,
        describe_snapshot(snapshot),
    )
    detail = "; ".join(verdict.reasons)
    try:
        if verdict.level in ("warning", "critical"):
            logger.warning("%s (%s)", message, detail)
        elif verdict.level == "ok":
            logger.info(message)
        else:
            logger.debug("%s (%s)", message, detail)
    except Exception:  # pragma: no cover - logging must never break startup
        pass
    return verdict


# ---------------------------------------------------------------------------
# OOM score adjustment
# ---------------------------------------------------------------------------


def _proc_path(pid: int, name: str) -> Path:
    return Path("/proc") / str(int(pid)) / name


def read_oom_score_adj(pid: int) -> int | None:
    """Current ``/proc/<pid>/oom_score_adj``, or ``None`` when unreadable."""
    try:
        return int((_proc_path(pid, "oom_score_adj")).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def apply_oom_score_adj(pid: int, value: int) -> tuple[int | None, str | None]:
    """Set ``/proc/<pid>/oom_score_adj`` and report what the kernel accepted.

    Returns ``(applied, error)``: *applied* is the value read back after the
    write (which is what the OOM killer will actually use, and may differ from
    *value* when the kernel clamps it), and *error* is a human-readable reason
    when the write was refused.

    Refusal is expected and harmless for negative *value* on an unprivileged
    process: raising a score is always allowed, lowering it below 0 requires
    ``CAP_SYS_RESOURCE``. Callers keep the current score in that case.
    """
    clamped = max(OOM_SCORE_ADJ_MIN, min(OOM_SCORE_ADJ_MAX, int(value)))
    path = _proc_path(pid, "oom_score_adj")
    try:
        path.write_text("%d\n" % clamped, encoding="utf-8")
        error = None
    except OSError as exc:
        error = "%s: %s" % (type(exc).__name__, exc.strerror or exc)
    return read_oom_score_adj(pid), error


def _server_cfg(config) -> dict:
    if isinstance(config, dict):
        section = config.get("server", {})
        if isinstance(section, dict):
            return section
    return {}


def protect_proxy_process(config=None, logger=None) -> int | None:
    """Apply the configured ``oom_score_adj`` to this proxy process.

    Best-effort: when the kernel refuses the negative value (unprivileged
    deployment) the current score is kept and a WARNING explains the remedy
    (install the systemd unit, which sets ``OOMScoreAdjust=-900`` as root, or
    grant ``CAP_SYS_RESOURCE``). Returns the applied score, or ``None`` when it
    could not be read.
    """
    configured = _server_cfg(config).get("oom_score_adj", PROXY_OOM_SCORE_ADJ)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = PROXY_OOM_SCORE_ADJ
    applied, error = apply_oom_score_adj(os.getpid(), configured)
    try:
        if logger is not None:
            if applied is None:
                logger.warning(
                    "Could not read oom_score_adj after setting %s; OOM protection "
                    "for the proxy is unverified",
                    configured,
                )
            elif error is not None and applied != configured:
                logger.warning(
                    "Could not lower oom_score_adj to %s (%s); keeping %s. "
                    "Negative scores need CAP_SYS_RESOURCE — run the proxy from the "
                    "systemd unit (OOMScoreAdjust=-900) or grant the capability. "
                    "The llama-server backend is still ranked above the proxy.",
                    configured,
                    error,
                    applied,
                )
            elif applied != configured:
                logger.warning(
                    "oom_score_adj applied as %s (requested %s); using the value "
                    "the kernel accepted",
                    applied,
                    configured,
                )
            else:
                logger.info(
                    "oom_score_adj=%s: the proxy outranks the model backend for the "
                    "OOM killer",
                    applied,
                )
    except Exception:  # pragma: no cover - logging must never break startup
        pass
    return applied


def protect_backend_process(proc, config=None, logger=None) -> int | None:
    """Raise the llama-server backend's ``oom_score_adj`` so it dies before the proxy.

    Returns the applied score, or ``None`` when the process handle has no pid or
    the score is unreadable. Never raises: a failed adjustment must not prevent
    a model from loading.
    """
    pid = getattr(proc, "pid", None)
    if not pid:
        return None
    configured = _server_cfg(config).get(
        "llama_server_oom_score_adj", LLAMA_SERVER_OOM_SCORE_ADJ
    )
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = LLAMA_SERVER_OOM_SCORE_ADJ
    applied, error = apply_oom_score_adj(int(pid), configured)
    try:
        if logger is not None and error is not None:
            logger.warning(
                "Could not set oom_score_adj=%s on llama-server pid %s (%s); the "
                "backend is not preferentially killable",
                configured,
                pid,
                error,
            )
        elif logger is not None and applied is not None and applied != configured:
            logger.warning(
                "llama-server pid %s oom_score_adj applied as %s (requested %s)",
                pid,
                applied,
                configured,
            )
    except Exception:  # pragma: no cover
        pass
    return applied


# ---------------------------------------------------------------------------
# Kernel OOM victim selection model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OomCandidate:
    """A process as the OOM killer sees it."""

    pid: int
    name: str
    rss_bytes: int
    oom_score_adj: int

    @property
    def killable(self) -> bool:
        """``oom_score_adj == -1000`` marks a process as never-killable."""
        return self.oom_score_adj > OOM_SCORE_ADJ_MIN


def oom_badness(candidate: OomCandidate, total_bytes: int) -> float | None:
    """Approximate the kernel's ``oom_badness()`` score for *candidate*.

    The kernel scales resident memory to 0..1000 permille of RAM and then scales
    that by ``(1000 + oom_score_adj) / 1000``. Returns ``None`` for a process
    that opted out with ``oom_score_adj == -1000`` (the kernel treats it as not
    killable). This is the model the 2026-11-02 event is reconstructed with:
    the proxy held 3.7 GiB RSS and llama-server 17.1 GiB on a 124 GiB host, both
    at ``oom_score_adj`` 0, so the backend scored ~4.6x the proxy — and a
    *global* OOM kept killing until both were gone.
    """
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if not candidate.killable:
        return None
    raw = 1000.0 * max(0, candidate.rss_bytes) / total_bytes
    adjusted = raw * (1000 + candidate.oom_score_adj) / 1000.0
    return max(0.0, adjusted)


def rank_oom_victims(
    candidates, total_bytes: int
) -> list[tuple[OomCandidate, float]]:
    """Return killable *candidates* ordered by descending badness (first victim first)."""
    scored = []
    for candidate in candidates:
        score = oom_badness(candidate, total_bytes)
        if score is None:
            continue
        scored.append((candidate, score))
    scored.sort(key=lambda item: (item[1], item[0].rss_bytes), reverse=True)
    return scored


def select_oom_victim(candidates, total_bytes: int) -> OomCandidate | None:
    """The process the kernel would kill first, or ``None`` when none is killable."""
    ranked = rank_oom_victims(candidates, total_bytes)
    return ranked[0][0] if ranked else None


# ---------------------------------------------------------------------------
# Recovery measurement (AC2)
# ---------------------------------------------------------------------------

#: A log gap at or above this is reported at WARNING: the previous process
#: stopped writing well before this one started, i.e. the proxy was down and
#: only a supervisor (systemd ``Restart=always``) or an operator brought it back.
STARTUP_GAP_WARNING_SECONDS = 300


def measure_startup_gap(log_dir, now: float | None = None) -> dict | None:
    """Measure how long ago the previous proxy process last wrote its log.

    ``setup_logging()`` renames a non-empty ``proxy.log`` aside to
    ``proxy.log.YYYY-MM-DD_HHMMSS`` at startup, so the newest archive's mtime is
    the last write of the previous run. The difference to *now* is the outage
    window the restart closed — the measured recovery interval for an
    involuntary death (SIGKILL/OOM), recorded in the new log so the next
    incident is not another ~10 minute silent gap with no figure attached.

    Returns ``{"gap_seconds", "last_write", "source"}`` or ``None`` when no
    previous log exists. Never raises.
    """
    if log_dir is None:
        return None
    directory = Path(log_dir)
    candidates: list[Path] = []
    try:
        candidates = [p for p in directory.glob("proxy.log.*") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        try:
            live = directory / "proxy.log"
            if live.is_file() and live.stat().st_size > 0:
                candidates = [live]
        except OSError:
            return None
    if not candidates:
        return None
    try:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        last_write = newest.stat().st_mtime
    except OSError:
        return None
    current = time.time() if now is None else float(now)
    return {
        "gap_seconds": max(0.0, current - last_write),
        "last_write": datetime.fromtimestamp(last_write, tz=UTC).isoformat(),
        "source": newest.name,
    }


def log_startup_gap(logger, log_dir, now: float | None = None) -> dict | None:
    """Log the measured outage window closed by this start (AC2). Never raises."""
    measured = measure_startup_gap(log_dir, now=now)
    try:
        if measured is None:
            if logger is not None:
                logger.debug(
                    "Startup gap: no previous proxy.log found; recovery interval "
                    "unknown (first start or a fresh log directory)"
                )
            return None
        message = (
            "Recovery: previous proxy log write %s (%s), %.0fs before this start"
            % (
                measured["last_write"],
                measured["source"],
                measured["gap_seconds"],
            )
        )
        if logger is not None:
            if measured["gap_seconds"] >= STARTUP_GAP_WARNING_SECONDS:
                logger.warning(
                    "%s — the proxy was down until this start; confirm the "
                    "supervisor (systemd Restart=always) is active so an "
                    "involuntary SIGKILL/OOM is recovered automatically",
                    message,
                )
            else:
                logger.info(message)
    except Exception:  # pragma: no cover - logging must never break startup
        pass
    return measured
