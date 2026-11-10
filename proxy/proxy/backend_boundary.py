"""
Supervision-boundary report for the llama-server backend (LP-0MW3PLK5M002YPOA).

The backend is killed by a *stray* ``SIGKILL`` sweep. The class-A attribution
child (LP-0MW3PLEH8008O4I7) named the sender — this repository's own
``proxy/scripts/start-proxy.sh --restart`` run from its tests, whose ``fuser -k
<port>`` phase selects by **port** and therefore cannot be sandboxed by a
``pgrep`` stub — and the decoy experiment in
``docs/death-attribution-LP-0MW3K7WYV001XLL1.md`` §9 proved the sweep does
**not** select ``argv[0]``. Name-based hardening is therefore not enough: the
durable answer is a **boundary** the sweep cannot cross.

This module makes that boundary *observable*. The deployed boundary is the
systemd user unit ``llama-server.service`` described in
``docs/systemd/llama-server.service`` and
``docs/dev/backend-supervision-boundary-runbook.md``:

- **cgroup** — the backend lives in ``…/user@<uid>.service/llama-server.service``,
  disjoint from the proxy's login-session scope, so a systemd-level
  ``KillMode=control-group`` (or ``systemctl --user stop``) targeted at the
  proxy cannot reach it;
- **process group / session** — the unit's ``ExecStart`` runs
  ``start-llama.sh`` as the unit's main process, so the router and its model
  workers are one session/PGID, distinct from the proxy's;
- **supervision** — ``Restart=always`` + ``RestartSec`` means even a kill that
  *does* land (a direct pid kill, or a port-selected ``fuser -k``) heals in
  seconds instead of stranding the backend (on 2026-11-05 the proxy's own spawn
  failed on a missing shared library and the router stayed down for hours).

The report (:func:`check_supervision_boundary`) is pure ``/proc`` +
``systemctl`` reading and never signals anything. It is **advisory**, not
fail-closed: an unmanaged backend (spawned inline by the proxy, the historical
coupling) is reported as ``state: "coupled"`` — a real immunity gap worth
seeing — but it is not an error, because that deployment still works.

Victim-side *signal* immunity is the companion guard
:mod:`proxy.backend_isolation` (disjoint PGIDs; the kill paths fail closed).
This module adds the wider boundary the PGID guard cannot express: the cgroup,
the unit name, and whether the backend is actually supervised by it.
"""

from __future__ import annotations

import os
import subprocess

#: The systemd user unit that owns the boundary (docs/systemd/llama-server.service).
BOUNDARY_UNIT_NAME = "llama-server.service"

#: How many seconds a ``systemctl`` probe may take before we give up (advisory
#: only — a slow probe must never slow /health down).
SYSTEMCTL_PROBE_TIMEOUT_SECONDS = 2.0


# ---------------------------------------------------------------------------
# /proc helpers (fail-open: unreadable => None)
# ---------------------------------------------------------------------------


def _read_cgroup(pid: int) -> str | None:
    """The cgroup v2 path of *pid* (first unified line), or None."""
    try:
        with open(f"/proc/{pid}/cgroup") as fh:
            for line in fh:
                # cgroup v2: "0::/path"; v1: "N:controller:/path"
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    return parts[2]
            return None
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None


def _pgid_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _sid_of(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _unit_scope_report() -> dict:
    """Whether ``llama-server.service`` is loaded, active and enabled.

    Best-effort and bounded: a missing ``systemctl`` (non-systemd host, test
    sandbox) degrades to ``{"available": False}``. Never raises.
    """
    report: dict = {"available": False, "unit": BOUNDARY_UNIT_NAME}
    try:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", BOUNDARY_UNIT_NAME],
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_PROBE_TIMEOUT_SECONDS,
        )
        report["available"] = True
        report["active"] = active.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return report
    try:
        enabled = subprocess.run(
            ["systemctl", "--user", "is-enabled", BOUNDARY_UNIT_NAME],
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_PROBE_TIMEOUT_SECONDS,
        )
        report["enabled"] = enabled.stdout.strip() == "enabled"
    except (subprocess.SubprocessError, OSError):
        report["enabled"] = None
    return report


# ---------------------------------------------------------------------------
# The boundary check
# ---------------------------------------------------------------------------


def check_supervision_boundary(srv, llama_pids: list[int] | None = None) -> dict:
    """Report whether the llama-server backend sits in its own boundary.

    The verdict (:attr:`state`):

    - ``"isolated"`` — the backend is live and its cgroup/unit, or its
      process group and session, are provably disjoint from the proxy's. A
      sweep of the proxy's group/scope cannot reach it.
    - ``"coupled"`` — the backend shares the proxy's cgroup *and* process
      group/session: the historical inline spawn. A stray sweep of the proxy
      reaches it; this is an immunity gap, reported (not an error).
    - ``"unknown"`` — the backend is not running or not resolvable; there is
      nothing to assert. Fail-open.

    ``violation`` mirrors ``state == "coupled"`` so a consumer can treat it
    like :func:`proxy.backend_isolation.check_backend_isolation`'s flag, and
    ``reason`` names the offenders.

    Never raises: every resolution failure degrades to ``unknown`` or skips
    that pid. Read-only — never signals a process.
    """
    report = {
        "guard": "supervision boundary (own systemd unit/cgroup + own PGID/SID)",
        "state": "unknown",
        "violation": False,
        "unit": BOUNDARY_UNIT_NAME,
        "proxy": {"pid": None, "cgroup": None, "pgid": None, "sid": None},
        "backend": {"pids": [], "cgroups": [], "pgids": [], "sids": []},
        "shared_cgroups": [],
        "shared_pgids": [],
        "reason": None,
    }
    try:
        proxy_pid = os.getpid()
        report["proxy"] = {
            "pid": proxy_pid,
            "cgroup": _read_cgroup(proxy_pid),
            "pgid": _pgid_of(proxy_pid),
            "sid": _sid_of(proxy_pid),
        }

        pids = list(llama_pids or [])
        if not pids:
            pids = _discover_llama_pids(srv)
        pids = sorted({p for p in pids if isinstance(p, int) and p > 1})

        live = [p for p in pids if _pid_alive(p)]
        report["backend"]["pids"] = live
        report["unit_scope"] = _unit_scope_report()
        if not live:
            return report

        report["backend"]["cgroups"] = sorted(
            {c for c in (_read_cgroup(p) for p in live) if c}
        )
        report["backend"]["pgids"] = sorted(
            {g for g in (_pgid_of(p) for p in live) if g}
        )
        report["backend"]["sids"] = sorted(
            {s for s in (_sid_of(p) for p in live) if s}
        )

        proxy_cgroup = report["proxy"]["cgroup"]
        proxy_pgid = report["proxy"]["pgid"]
        proxy_sid = report["proxy"]["sid"]

        shared_cgroups = (
            [proxy_cgroup]
            if proxy_cgroup and proxy_cgroup in report["backend"]["cgroups"]
            else []
        )
        shared_pgids = sorted(
            {g for g in report["backend"]["pgids"] if g and g == proxy_pgid}
        )
        shares_session = bool(proxy_sid) and proxy_sid in report["backend"]["sids"]
        report["shared_cgroups"] = shared_cgroups
        report["shared_pgids"] = shared_pgids

        # The boundary holds as soon as ONE axis separates the backend from
        # the proxy: a disjoint PGID already blocks a `kill -<proxy-pgid>`
        # sweep, and a distinct cgroup additionally blocks a systemd
        # scope-level kill. Only when every axis is shared (the historical
        # inline spawn) is the backend actually inside the proxy's kill set.
        pgids_disjoint = not shared_pgids and not shares_session
        cgroups_disjoint = not shared_cgroups

        if pgids_disjoint or cgroups_disjoint:
            report["state"] = "isolated"
        else:
            report["state"] = "coupled"
            report["reason"] = _coupled_reason(shared_cgroups, shared_pgids)

        report["violation"] = report["state"] == "coupled"
        return report
    except Exception as exc:  # pragma: no cover - defensive
        report["reason"] = f"check_failed: {exc}"
        return report


def _coupled_reason(shared_cgroups: list[str], shared_pgids: list[int]) -> str:
    parts = []
    if shared_pgids:
        parts.append(f"shared process group {shared_pgids}")
    if shared_cgroups:
        parts.append(f"shared cgroup {shared_cgroups}")
    return "; ".join(parts) or "backend not separated from the proxy"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:  # pragma: no cover - defensive
        return False


def _discover_llama_pids(srv) -> list[int]:
    """Resolve live llama-server pids (handle first, then by port)."""
    pids: list[int] = []
    try:
        proc = getattr(srv, "llama_process", None)
        pid = getattr(proc, "pid", None)
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
            pids.append(pid)
    except Exception:
        pass
    if not pids:
        try:
            from proxy.backend_rss import discover_llama_pids

            server_cfg = (
                srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
            )
            port = int(server_cfg.get("llama_server_port", 8080) or 8080)
            pids.extend(p for p in discover_llama_pids(port) if p > 1)
        except Exception:
            pass
    return pids
