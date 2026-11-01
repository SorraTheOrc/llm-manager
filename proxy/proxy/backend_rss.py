"""Resolve llama-server resident memory without a host process handle.

LP-0MVO9F877008ZCP7: ``llama_process_rss_bytes`` / ``llama_model_rss_bytes``
were published as ``0`` on every scrape because ``/admin/metrics`` read RSS
only via ``srv.llama_process.pid`` (a host ``subprocess.Popen`` handle). Three
real-world failure modes made that handle unavailable or misleading:

1. **Podman-backed deployment**: llama-server runs inside a container started
   by ``scripts/podman_start_llama.sh`` (``PODMAN_LC_CONTAINER``/``llama``), so
   ``srv.llama_process`` is the host-side ``podman exec`` wrapper whose RSS is
   the wrapper's own (tiny), not the backend's.
2. **Adopted server across a proxy restart**: ``_AdoptedLlamaProcess`` carries
   ``pid = 0``, so the existing ``if pid:`` guard was false and RSS stayed
   ``None`` (verified live 2026-10-26).
3. **Router-mode children**: model memory lives in *child* processes of the
   router (measured ~44 GB across two children vs ~0.2 GB in the router
   process itself), so reading only the top-level process under-reports by
   ~99%.

Approach (additive — the host pid path is kept):

- When a container is configured/auto-detected, resolve the container init
  PID via ``podman inspect -f '{{.State.Pid}}' <name>`` and sum the RSS of that
  process tree (exact, matches ``ps``), with the cgroup ``memory.current`` as a
  fallback (the same number ``podman stats`` reports; differs from pure RSS by
  the page-cache share, which ``--no-mmap`` keeps tiny).
- Otherwise, resolve the llama-server process tree on the host: the tracked
  ``llama_process.pid`` when it is a genuine backend process (never when it is
  a ``podman`` wrapper), else discover router pids by scanning ``/proc`` for
  ``llama-server`` processes listening on the configured ``llama_server_port``.
  RSS is the sum over the process tree, so the router-child case is covered.

Every failure is fail-soft (the scrape still succeeds) and **logged once per
reason**, not per scrape. On a transient resolution failure the last
known-good value is kept instead of zeroing the gauge, so an alert on the
series cannot be silently disabled by a single failed lookup; only a backend
that has *never* resolved keeps the registered ``0``.

Per-model attribution stays the documented equal-split heuristic in
``proxy.metrics.update_metrics`` (labels sum to the process total).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_REFRESH_SECONDS = 5.0
"""Max age of a cached RSS value before a scrape re-resolves it (seconds)."""

DEFAULT_MODELS_REFRESH_SECONDS = 30.0
"""Max age of the cached loaded-model list before a scrape re-lists it."""

PODMAN_TIMEOUT_SECONDS = 2.0
"""Per-``podman inspect`` subprocess timeout (a hung runtime must not block a scrape)."""

DEFAULT_CONTAINER_NAME = "llama"
"""Fallback container name matching ``scripts/podman_start_llama.sh``."""

_ENABLED = os.environ.get("LLAMA_RSS_DISABLED", "") not in {"1", "true", "True", "yes"}

_state: dict = {
    "at": 0.0,          # monotonic time of the last RSS resolution
    "rss": None,        # last known-good RSS bytes (None = never resolved)
    "source": None,     # resolution strategy label, e.g. "container-process-tree"
    "models": (),       # last known loaded-model ids
    "models_at": 0.0,   # monotonic time of the last successful model list
    "models_miss_at": 0.0,  # monotonic time of the last failed model list
}
_warned: set[str] = set()
"""Reasons already logged this process lifetime (warn-once throttle)."""

_LOCK = threading.Lock()
"""Serialises RSS resolution so a burst of scrapes cannot stack subprocesses."""

logger = logging.getLogger("llama-proxy")


def reset_state() -> None:
    """Clear the cache and warn-once set (test isolation / restart hook)."""
    with _LOCK:
        _state.update(at=0.0, rss=None, source=None, models=(), models_at=0.0, models_miss_at=0.0)
        _warned.clear()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _server_config(srv) -> dict:
    try:
        cfg = getattr(srv, "config", None) or {}
    except Exception:
        cfg = {}
    server = cfg.get("server") if isinstance(cfg, dict) else None
    return server if isinstance(server, dict) else {}


def _refresh_seconds(srv) -> float:
    try:
        return max(1.0, float(_server_config(srv).get("llama_rss_refresh_seconds") or DEFAULT_REFRESH_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_REFRESH_SECONDS


def _models_refresh_seconds(srv) -> float:
    try:
        return max(1.0, float(_server_config(srv).get("llama_rss_models_refresh_seconds") or DEFAULT_MODELS_REFRESH_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_MODELS_REFRESH_SECONDS


def _llama_port(srv) -> int:
    try:
        return int(_server_config(srv).get("llama_server_port") or 8080)
    except (TypeError, ValueError):
        return 8080


def _router_mode(srv) -> bool:
    return bool(_server_config(srv).get("llama_router_mode", False))


def _log_warn_once(srv, key: str, message: str, *args) -> None:
    """Log *message* at WARNING level at most once per *key* per process lifetime."""
    if key in _warned:
        return
    _warned.add(key)
    log = getattr(srv, "logger", None) or logger
    try:
        log.warning(message, *args)
    except Exception:
        try:
            log.warning("%s", message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# /proc readers (no psutil dependency; works in and out of containers)
# ---------------------------------------------------------------------------


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError):
        return 4096


def _proc_stat(pid: int) -> tuple[str, int] | None:
    """Return ``(state, ppid)`` for *pid*, or None when the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    rest = raw[close + 2:].split()
    if len(rest) < 2:
        return None
    try:
        return rest[0], int(rest[1])
    except ValueError:
        return None


def _proc_rss_bytes(pid: int) -> int | None:
    """Resident bytes of *pid* from ``/proc/<pid>/statm`` (field 2, pages)."""
    try:
        raw = Path(f"/proc/{pid}/statm").read_text().split()
    except OSError:
        return None
    if len(raw) < 2:
        return None
    try:
        return int(raw[1]) * _page_size()
    except ValueError:
        return None


def _cmdline(pid: int) -> str:
    """Join the NUL-separated args of *pid* into a single string ("" if gone)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _proc_pids() -> list[int]:
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def _process_table() -> dict[int, list[int]]:
    """Map ppid -> [pid, ...] for every visible process (one pass over /proc)."""
    table: dict[int, list[int]] = {}
    for pid in _proc_pids():
        st = _proc_stat(pid)
        if st is None:
            continue
        table.setdefault(st[1], []).append(pid)
    return table


def _tree_pids(root_pids, table: dict[int, list[int]] | None = None) -> list[int]:
    """All pids reachable from *root_pids* (inclusive) via the children map."""
    table = table if table is not None else _process_table()
    seen: set[int] = set()
    stack = [int(p) for p in root_pids if int(p) > 0]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(table.get(pid, ()))
    return list(seen)


def _rss_of_pids(pids) -> int | None:
    total = 0
    found = False
    for pid in pids:
        rss = _proc_rss_bytes(pid)
        if rss is not None:
            total += rss
            found = True
    return total if found else None


def _tree_rss_bytes(root_pids, table: dict[int, list[int]] | None = None) -> int | None:
    """Sum RSS over the process trees rooted at *root_pids* (no double counting)."""
    return _rss_of_pids(_tree_pids(root_pids, table))


# ---------------------------------------------------------------------------
# Container resolution (podman)
# ---------------------------------------------------------------------------


def container_name(srv) -> str | None:
    """Resolve the llama-server container name, or None for host-direct.

    Priority: the ``server.llama_container_name`` config key, then the
    ``PODMAN_LC_CONTAINER`` environment variable (used by
    ``scripts/podman_start_llama.sh``), then an auto-detect where the
    configured start script is itself the podman wrapper (default name
    ``llama``). None means "no container is involved — use the host path".
    """
    cfg = _server_config(srv)
    name = cfg.get("llama_container_name") or os.environ.get("PODMAN_LC_CONTAINER")
    if name:
        name = str(name).strip()
        if name:
            return name
    script = str(cfg.get("llama_start_script") or "")
    if script and "podman" in Path(script).name.lower():
        return DEFAULT_CONTAINER_NAME
    return None


def container_init_pid(name: str) -> int | None:
    """Host PID of the container's init process via ``podman inspect``.

    Returns None when podman is unavailable, the container does not exist / is
    not running, or the inspect call fails or times out — the caller falls
    back to the host process path.
    """
    if not _ENABLED:
        return None
    podman = shutil.which("podman")
    if not podman:
        return None
    try:
        proc = subprocess.run(
            [podman, "inspect", "-f", "{{.State.Pid}}", name],
            capture_output=True, text=True, timeout=PODMAN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, TypeError):
        return None
    if proc.returncode != 0:
        return None
    try:
        pid = int(proc.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0
    return pid if pid > 0 else None


def cgroup_memory_current(
    pid: int, proc_root: str | Path = "/proc", sys_root: str | Path = "/sys/fs/cgroup"
) -> int | None:
    """Bytes from the cgroup v2 ``memory.current`` of *pid*, or None.

    The container's own cgroup (systemd driver: ``libpod-<id>.scope``) accounts
    every process in the container, so this is the same number ``podman stats``
    reports as memory usage. Differs from pure RSS by the page-cache share,
    which ``--no-mmap`` keeps tiny — well inside AC1's 5 % tolerance either way.

    ``proc_root``/``sys_root`` exist only for tests (tmp dirs).
    """
    try:
        lines = Path(proc_root) / str(pid) / "cgroup"
        lines = lines.read_text().splitlines()
    except OSError:
        return None
    path = None
    for line in lines:
        # cgroup v2 unified hierarchy line: "0::/machine.slice/libpod-x.scope"
        if line.startswith("0::"):
            path = line[3:].strip()
            break
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            path = parts[2].strip()
            break
    if not path:
        return None
    try:
        current = Path(sys_root) / path.lstrip("/") / "memory.current"
        return int(current.read_text().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Host process resolution
# ---------------------------------------------------------------------------


def _tracked_pid(srv) -> int | None:
    """Host pid owned by the proxy's spawned llama-server when usable.

    Returns None when the handle is missing, adopted with ``pid = 0``, dead,
    or (critically) when the process is a host-side ``podman exec`` wrapper —
    its RSS is the wrapper's own, not the backend's.
    """
    try:
        proc = getattr(srv, "llama_process", None)
        raw = getattr(proc, "pid", 0) or 0
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if getattr(proc, "adopted", False):
        return None
    if _proc_stat(pid) is None:
        return None
    if "podman" in _cmdline(pid):
        return None
    return pid


_HOST_PORT_RE_CACHE: dict[int, re.Pattern] = {}


def _port_pattern(port: int) -> re.Pattern:
    pattern = _HOST_PORT_RE_CACHE.get(port)
    if pattern is None:
        pattern = re.compile(rf"--port[= ]{port}(?:\s|$)")
        _HOST_PORT_RE_CACHE[port] = pattern
    return pattern


def discover_llama_pids(port: int | None = None) -> list[int]:
    """Pids of host ``llama-server`` processes listening on *port*.

    Works for host-direct and container deployments alike (container processes
    are visible in the host pid namespace). The port filter targets the router
    (started with ``--port <llama_server_port>``) and never matches a child
    (children start with a dynamic ``--port 0``).
    """
    pattern = _port_pattern(port) if port else None
    found = []
    for pid in _proc_pids():
        cmd = _cmdline(pid)
        if "llama-server" not in cmd:
            continue
        if pattern and pattern.search(cmd):
            found.append(pid)
    return found


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------


def _container_rss(srv, name: str) -> tuple[int, str] | None:
    pid = container_init_pid(name)
    if not pid:
        _log_warn_once(
            srv, f"container-not-running:{name}",
            "llama RSS: podman container %r is not running (or podman is "
            "unavailable); falling back to the host process tree",
            name,
        )
        return None
    rss = _tree_rss_bytes([pid])
    if rss:
        return rss, "container-process-tree"
    mem = cgroup_memory_current(pid)
    if mem:
        return mem, "container-cgroup"
    _log_warn_once(
        srv, f"container-no-memory:{name}",
        "llama RSS: container %r (init pid %d) exposed neither a readable "
        "process tree nor cgroup memory.current; publishing last known value",
        name, pid,
    )
    return None


def _host_rss(srv) -> tuple[int, str] | None:
    tracked = _tracked_pid(srv)
    table = _process_table()
    if tracked:
        rss = _tree_rss_bytes([tracked], table)
        if rss:
            return rss, "host-process-tree"
        _log_warn_once(
            srv, "host-unreadable",
            "llama RSS: tracked llama-server pid %d has no readable RSS; "
            "trying port discovery", tracked,
        )
    pids = discover_llama_pids(_llama_port(srv))
    if not pids:
        return None
    rss = _tree_rss_bytes(pids, table)
    if rss:
        return rss, "host-process-tree"
    return None


def _resolve_once(srv) -> tuple[int | None, str | None]:
    """Resolve RSS bytes and the strategy label used (or (None, None))."""
    if not _ENABLED:
        return None, None
    name = container_name(srv)
    if name:
        result = _container_rss(srv, name)
        if result is not None:
            return result
    result = _host_rss(srv)
    if result is not None:
        return result
    _log_warn_once(
        srv, "no-runtime",
        "llama RSS: no container or host llama-server process could be "
        "resolved; the gauge keeps its last value (0 until the first success)",
    )
    return None, None


# ---------------------------------------------------------------------------
# Public API used by the /metrics and /admin/metrics handlers
# ---------------------------------------------------------------------------


def last_source() -> str | None:
    """Label of the strategy that produced the current gauge value (diagnostics)."""
    return _state.get("source")


def update_gauges(srv, loaded_models=None, force: bool = False) -> int | None:
    """Refresh the RSS gauges and return the published process RSS.

    TTL-bounded by ``llama_rss_refresh_seconds`` so a Prometheus scrape of
    ``/metrics`` (which never calls ``/admin/metrics``) keeps the series fresh
    without re-running ``podman``/``/proc`` scans on every single scrape. On a
    transient resolution failure the last known-good value is kept (never
    zeroed, so an alert cannot be silently disabled); only a backend that has
    never resolved publishes ``0``.
    """
    with _LOCK:
        now = time.monotonic()
        ttl = _refresh_seconds(srv)
        if loaded_models is not None:
            models = list(loaded_models)
            if models:
                _state["models"] = tuple(models)
                _state["models_at"] = now
        else:
            models = list(_state["models"])

        stale = _state["at"] == 0.0 or (now - _state["at"]) >= ttl
        if force or stale:
            try:
                rss, source = _resolve_once(srv)
            except Exception:
                rss, source = None, None
            if rss is not None:
                _state.update(at=now, rss=rss, source=source)
            else:
                # Keep the previous (possibly None) value and remember we tried.
                rss = _state["rss"]
                _state.update(at=now, source=source)
        else:
            rss, source = _state["rss"], _state["source"]

    try:
        import proxy.metrics as metrics

        metrics.update_metrics(rss, models or None)
    except Exception:
        pass
    return rss


async def ensure_models(srv, force: bool = False) -> list[str]:
    """Return the loaded-model ids, refreshing from the router at most every
    ``llama_rss_models_refresh_seconds``.

    ``/metrics`` has no other source of loaded models (it must not call
    ``/admin/metrics``), so on the first scrape after startup it lists the
    router's models (fail-soft: a failed list reuses the last known set and is
    only retried after the TTL, never per scrape). Non-router mode attributes
    the single ``current_model``.
    """
    now = time.monotonic()
    ttl = _models_refresh_seconds(srv)
    cached = list(_state["models"])
    if not force and cached and (now - _state["models_at"]) < ttl:
        return cached
    if not force and _state["models_miss_at"] and (now - _state["models_miss_at"]) < ttl:
        return cached
    models: list[str] = []
    try:
        if _router_mode(srv):
            router_models = await srv.router_list_models()
            models = list(srv._extract_router_model_ids(router_models) or [])
        else:
            current = getattr(srv, "current_model", None)
            models = [str(current)] if current else []
    except Exception:
        models = []
    if models:
        _state["models"] = tuple(models)
        _state["models_at"] = now
    else:
        # Negative cache: do not hammer the router on every scrape after a
        # transient failure.
        _state["models_miss_at"] = now
    return models or cached
