
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
  see GTT, ``gpu_busy_percent`` is useless on this APU (a two-state detector
  whose step sits at ~30 W, below the idle-with-display band — see
  ``monitoring/README.md``, "Why not amd-smi, rocm-smi or amdtop?"), and the
  only evidence left was a kernel log line.

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

The score must land on the **llama-server process**, not on the wrapper the
proxy spawned (LP-0MW3K7XJ30027RV6). ``start-llama.sh`` runs ``llama-server`` as
a foreground child, so a write to the wrapper's pid after that fork is never
inherited: measured live on 2026-11-05, the router and its model child both sat
at ``oom_score_adj=0`` while the configuration asked for ``800``. The
protection therefore resolves the process **tree** (:func:`process_tree`,
:func:`backend_pids`), scores every ``llama-server`` process in it
(:func:`apply_backend_oom_protection`) and can re-assert — and verify — the
ranking later (:func:`reapply_backend_oom_protection`,
:func:`inspect_backend_oom_protection`), which also covers the model children
the router forks when it loads a model. ``oom_score_adj`` is copied by the
kernel at fork time, so scoring the router first makes every later child inherit
the raised value.

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
                f"MemAvailable {available_fraction * 100:.1f}% of MemTotal "
                f"(critical threshold {available_critical_fraction * 100:.0f}%)"
            )
        elif available_fraction <= available_warn_fraction:
            warning.append(
                f"MemAvailable {available_fraction * 100:.1f}% of MemTotal "
                f"(warning threshold {available_warn_fraction * 100:.0f}%)"
            )

    gtt_fraction = snapshot.gtt_host_fraction
    if gtt_fraction is not None:
        if gtt_fraction >= gtt_critical_fraction:
            critical.append(
                f"GPU GTT {gtt_fraction * 100:.1f}% of physical RAM "
                f"(critical threshold {gtt_critical_fraction * 100:.0f}%)"
            )
        elif gtt_fraction >= gtt_warn_fraction:
            warning.append(
                f"GPU GTT {gtt_fraction * 100:.1f}% of physical RAM "
                f"(warning threshold {gtt_warn_fraction * 100:.0f}%)"
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
        return "unknown" if value is None else f"{value / (1024**3):.1f} GiB"

    return (
        f"MemAvailable {_gib(snapshot.mem_available_bytes)} of "
        f"{_gib(snapshot.mem_total_bytes)}, GPU GTT "
        f"{_gib(snapshot.gtt_used_bytes)} of {_gib(snapshot.gtt_total_bytes)}"
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
    message = (
        f"Host memory pressure [{context}]: {verdict.level} — "
        f"{describe_snapshot(snapshot)}"
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
        path.write_text(f"{clamped}\n", encoding="utf-8")
        error = None
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc.strerror or exc}"
    return read_oom_score_adj(pid), error


# ---------------------------------------------------------------------------
# Process-tree resolution: the score belongs on llama-server, not its wrapper
# ---------------------------------------------------------------------------

DEFAULT_PROC_ROOT = Path("/proc")

#: Kernel ``comm`` (and first ``cmdline`` field) of the backend binary. Truncated
#: to 15 characters in ``comm``, which still contains this whole marker.
LLAMA_SERVER_MARKER = "llama-server"


@dataclass(frozen=True)
class ProcessEntry:
    """One ``/proc/<pid>`` row: parent, kernel ``comm`` and full command line."""

    pid: int
    ppid: int
    comm: str
    cmdline: str

    @property
    def is_llama_server(self) -> bool:
        """True for a llama-server process, however it was launched.

        ``comm`` is the executable name (``llama-server``). A stand-in started
        with ``exec -a llama-server`` keeps its own ``comm`` but carries the
        name in ``argv[0]``. Only ``argv[0]`` is matched — never the whole
        command line — so a wrapper whose *script text* mentions llama-server
        (``bash -c '… llama-server …'``, ``podman run … llama-server``) is not
        mistaken for the backend. The container case is still covered: the tree
        walk reaches the container's ``llama-server`` process whose ``comm``
        matches.
        """
        if LLAMA_SERVER_MARKER in self.comm:
            return True
        return LLAMA_SERVER_MARKER in Path(self.argv0).name

    @property
    def argv0(self) -> str:
        """First ``cmdline`` field (the program name), or ``""`` when unknown."""
        return self.cmdline.split(" ", 1)[0] if self.cmdline else ""


def _parse_stat(raw: str) -> tuple[str, int] | None:
    """``(comm, ppid)`` from one ``/proc/<pid>/stat`` line.

    ``comm`` is wrapped in parentheses and the kernel does **not** escape it,
    so a process can legitimately be called ``foo) 1`` and a naive split on the
    first ``)`` would read the wrong fields. The outermost pair is therefore
    located with the first ``(`` and the last ``)``.
    """
    left = raw.find("(")
    right = raw.rfind(")")
    if left == -1 or right <= left:
        return None
    fields = raw[right + 1 :].split()
    # fields[0] is the state, fields[1] the parent pid.
    if len(fields) < 2:
        return None
    try:
        return raw[left + 1 : right], int(fields[1])
    except ValueError:
        return None


def read_process_table(proc_root=None) -> dict[int, ProcessEntry]:
    """Read ``{pid: ProcessEntry}`` for every process this user can inspect.

    Processes that exit mid-read are skipped: the table is a best-effort
    snapshot and a vanished pid must never raise.
    """
    root = Path(proc_root) if proc_root is not None else DEFAULT_PROC_ROOT
    table: dict[int, ProcessEntry] = {}
    try:
        candidates = list(root.iterdir())
    except OSError:
        return table
    for candidate in candidates:
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        try:
            stat = (candidate / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = _parse_stat(stat)
        if parsed is None:
            continue
        comm, ppid = parsed
        try:
            raw_cmdline = (candidate / "cmdline").read_bytes()
        except OSError:
            raw_cmdline = b""
        table[pid] = ProcessEntry(
            pid=pid,
            ppid=ppid,
            comm=comm,
            cmdline=raw_cmdline.replace(b"\0", b" ").decode("utf-8", "replace").strip(),
        )
    return table


def process_tree(root_pids, table=None, proc_root=None) -> list[int]:
    """*root_pids* followed by every descendant, breadth-first.

    ``start-llama.sh``/podman/proxy layers mean the process the proxy holds a
    handle for is usually an ancestor of the real backend, so scoring only the
    handle leaves the backend (and its router children) unprotected.
    """
    known = table if table is not None else read_process_table(proc_root)
    roots: list[int] = []
    for pid in root_pids:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid > 0 and pid not in roots:
            roots.append(pid)
    children: dict[int, list[int]] = {}
    for pid in sorted(known):
        children.setdefault(known[pid].ppid, []).append(pid)
    ordered: list[int] = []
    queue = list(roots)
    while queue:
        pid = queue.pop(0)
        if pid in ordered:
            continue
        ordered.append(pid)
        queue.extend(children.get(pid, ()))
    return ordered


def backend_pids(root_pids, table=None, proc_root=None) -> list[int]:
    """Pids of the ``llama-server`` processes among *root_pids* and their trees."""
    known = table if table is not None else read_process_table(proc_root)
    return [
        pid
        for pid in process_tree(root_pids, table=known)
        if (entry := known.get(pid)) is not None and entry.is_llama_server
    ]


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


def _configured_backend_score(config) -> int:
    """``server.llama_server_oom_score_adj``, or the module default."""
    configured = _server_cfg(config).get(
        "llama_server_oom_score_adj", LLAMA_SERVER_OOM_SCORE_ADJ
    )
    try:
        return int(configured)
    except (TypeError, ValueError):
        return LLAMA_SERVER_OOM_SCORE_ADJ


def _coerce_pid(value) -> int | None:
    """A positive pid from *value*, else ``None`` (``_AdoptedLlamaProcess`` is 0)."""
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


@dataclass(frozen=True)
class OomScoreWrite:
    """What one ``/proc/<pid>/oom_score_adj`` write (or read) produced."""

    pid: int
    comm: str
    requested: int
    applied: int | None
    error: str | None = None
    #: True for the ``llama-server`` process itself (as opposed to a wrapper or
    #: start-script ancestor scored so later forks inherit the raised value).
    is_backend: bool = False

    @property
    def ok(self) -> bool:
        """The kernel accepted the requested value (a read-back comparison)."""
        return self.error is None and self.applied == self.requested


@dataclass(frozen=True)
class BackendOomReport:
    """Verification result for one backend process tree.

    ``gap`` is the documented explanation when the intended ranking is *not*
    live (no backend in the tree, a refused write, or a process that vanished),
    so an unverified backend is never reported as protected.
    """

    configured: int
    rows: tuple[OomScoreWrite, ...] = ()
    gap: str | None = None

    @property
    def backend_rows(self) -> tuple[OomScoreWrite, ...]:
        return tuple(row for row in self.rows if row.is_backend)

    @property
    def protected(self) -> bool:
        return self.gap is None and bool(self.backend_rows) and all(
            row.ok for row in self.backend_rows
        )

    def as_dict(self) -> dict:
        """JSON-friendly view (the runbook tool and the watchdog both use it)."""
        return {
            "configured": self.configured,
            "protected": self.protected,
            "gap": self.gap,
            "rows": [
                {
                    "pid": row.pid,
                    "comm": row.comm,
                    "requested": row.requested,
                    "applied": row.applied,
                    "error": row.error,
                    "is_backend": row.is_backend,
                }
                for row in self.rows
            ],
        }


#: State keys already reported this process lifetime (log-once throttle).
_REPORTED: set[str] = set()

#: Bound on :data:`_REPORTED`; a long-lived proxy must not accumulate pid keys
#: forever, and re-reporting after a reset is at most one line per 1024 changes.
_REPORTED_MAX = 1024


def _log_once(logger, key: str, level: str, message: str, *args) -> None:
    """Log *message* at *level* once per *key*; never raises.

    One reconcile pass emits one line per state, not one per 30 s pass: a
    router child that appears later has a new pid and therefore a new key, so a
    real change is still reported.
    """
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    if len(_REPORTED) > _REPORTED_MAX:
        _REPORTED.clear()
    try:
        if logger is not None:
            getattr(logger, level, logger.warning)(message, *args)
    except Exception:  # pragma: no cover - logging must never break startup
        pass


def _is_gone_error(error: str | None) -> bool:
    """True when *error* means the process no longer exists (not a refusal)."""
    if not error:
        return False
    return any(
        marker in error
        for marker in ("No such file or directory", "ProcessLookupError", "ESRCH")
    )


def _backend_rows(root_pids, configured: int, *, write: bool) -> list[OomScoreWrite]:
    """Score (or read) every root and every ``llama-server`` in their trees."""
    table = read_process_table()
    roots = {
        pid
        for pid in (_coerce_pid(value) for value in root_pids)
        if pid is not None
    }
    rows: list[OomScoreWrite] = []
    for pid in process_tree(sorted(roots), table=table):
        entry = table.get(pid)
        is_backend = entry is not None and entry.is_llama_server
        if pid not in roots and not is_backend:
            # Scoring a transient helper (awk, mktemp) would be noise; only the
            # roots (so later forks inherit) and the backend itself matter.
            continue
        if write:
            applied, error = apply_oom_score_adj(pid, configured)
        else:
            applied, error = read_oom_score_adj(pid), None
        rows.append(
            OomScoreWrite(
                pid=pid,
                comm=entry.comm if entry is not None else "",
                requested=configured,
                applied=applied,
                error=error,
                is_backend=is_backend,
            )
        )
    return rows


def _build_report(
    rows: list[OomScoreWrite],
    configured: int,
    roots,
    *,
    logger=None,
    require_backend: bool = False,
) -> BackendOomReport:
    """Summarise *rows*, logging the gaps and the once-per-run verification."""
    gap: str | None = None
    backend_rows = [row for row in rows if row.is_backend]
    refused = [row for row in rows if row.error is not None and not _is_gone_error(row.error)]
    mismatched = [row for row in backend_rows if row.error is None and not row.ok]
    if not backend_rows:
        gap = (
            "no llama-server process in the tree of pid(s) "
            f"{', '.join(str(pid) for pid in sorted(roots)) or 'none'} — the backend "
            "is not running, its start script has not forked it yet, or it could "
            "not be discovered (an adopted backend whose port is unreadable); its "
            "oom_score_adj is unverified"
        )
    elif refused:
        gap = (
            f"could not set oom_score_adj={configured} on "
            + ", ".join(f"pid {row.pid} ({row.error})" for row in refused)
        )
    elif mismatched:
        gap = (
            "oom_score_adj on "
            + ", ".join(f"pid {row.pid}" for row in mismatched)
            + f" reads {', '.join(str(row.applied) for row in mismatched)}, expected "
            f"{configured} (the kernel is not holding the intended ranking)"
        )

    report = BackendOomReport(configured=configured, rows=tuple(rows), gap=gap)
    if gap is not None:
        if require_backend:
            _log_once(
                logger,
                f"backend-oom-gap:{gap}",
                "warning",
                "Backend OOM protection is not verified: %s. The model backend is "
                "not preferentially killable; reapply and verify by hand with "
                "`python3 scripts/verify_oom_protection.py --apply` (see "
                "docs/dev/oom-recovery-runbook.md), or restart the backend so the "
                "proxy scores it at spawn time.",
                gap,
            )
        return report
    for row in backend_rows:
        if row.error is None:
            _log_once(
                logger,
                f"backend-oom-ok:{row.pid}:{row.applied}",
                "info",
                "oom_score_adj=%s verified on llama-server pid %s (%s), above the "
                "proxy for the OOM killer",
                row.applied,
                row.pid,
                row.comm or "comm unreadable",
            )
    return report


def apply_backend_oom_protection(root_pids, configured: int) -> list[OomScoreWrite]:
    """Raise ``oom_score_adj`` on *root_pids* and every llama-server below them."""
    return _backend_rows(root_pids, int(configured), write=True)


def reapply_backend_oom_protection(
    root_pids, config=None, logger=None, *, require_backend: bool = False
) -> BackendOomReport:
    """Re-assert the backend ranking on a running tree (router and children).

    Idempotent and safe to call periodically: it walks the tree from the pids the
    caller knows about (the tracked spawn handle plus any port discovery), writes
    the configured score to every ``llama-server`` process it finds and verifies
    the result from ``/proc``. A backend that cannot be found or scored is
    reported as a :attr:`BackendOomReport.gap` — with *require_backend*, the gap
    is logged once together with the operator action — never silently passed.
    """
    configured = _configured_backend_score(config)
    rows = apply_backend_oom_protection(root_pids, configured)
    return _build_report(
        rows, configured, root_pids, logger=logger, require_backend=require_backend
    )


def inspect_backend_oom_protection(
    root_pids, config=None, *, require_backend: bool = False
) -> BackendOomReport:
    """Read-only verification of the backend ranking from ``/proc``."""
    configured = _configured_backend_score(config)
    rows = _backend_rows(root_pids, configured, write=False)
    return _build_report(rows, configured, root_pids, require_backend=require_backend)


def protect_backend_process(proc, config=None, logger=None) -> int | None:
    """Raise the llama-server backend's ``oom_score_adj`` so it dies before the proxy.

    The write goes to the process the caller spawned **and** to every
    ``llama-server`` process already below it, because ``start-llama.sh`` forks
    the backend before the proxy can reach ``/proc`` (LP-0MW3K7XJ30027RV6).
    Scoring the wrapper first is what makes a child forked *later* — a model
    loaded after the write — inherit the raised value.

    Returns the applied score of *proc*, or ``None`` when the process handle has
    no pid or the score is unreadable. Never raises: a failed adjustment must not
    prevent a model from loading.
    """
    pid = _coerce_pid(getattr(proc, "pid", None))
    if pid is None:
        return None
    configured = _configured_backend_score(config)
    rows = apply_backend_oom_protection([pid], configured)
    for row in rows:
        try:
            if logger is None:
                continue
            if row.error is not None:
                if not _is_gone_error(row.error):
                    logger.warning(
                        "Could not set oom_score_adj=%s on pid %s (%s); the backend "
                        "is not preferentially killable",
                        configured,
                        row.pid,
                        row.error,
                    )
                else:
                    logger.debug(
                        "pid %s exited before oom_score_adj=%s could be applied",
                        row.pid,
                        configured,
                    )
            elif row.applied is not None and row.applied != configured:
                logger.warning(
                    "pid %s oom_score_adj applied as %s (requested %s)",
                    row.pid,
                    row.applied,
                    configured,
                )
            elif row.is_backend:
                logger.info(
                    "llama-server pid %s oom_score_adj=%s (backend ranked above the "
                    "proxy for the OOM killer)",
                    row.pid,
                    row.applied,
                )
        except Exception:  # pragma: no cover - logging must never break startup
            pass
    root_row = next((row for row in rows if row.pid == pid), None)
    return root_row.applied if root_row is not None else None


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
            f"Recovery: previous proxy log write {measured['last_write']} "
            f"({measured['source']}), {measured['gap_seconds']:.0f}s before "
            "this start"
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
