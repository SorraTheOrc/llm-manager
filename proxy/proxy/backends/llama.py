"""
Llama Backend Module

Watchdog, health probing, and recovery logic specific to llama-server.

Uses a lazy server import (_srv()) and shared utilities from the parent
backend_health module.
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path

import httpx

from .. import death_capture as _death_capture

#: ``restart_reason`` label for llama-server deaths (see LP-0MW3K7WD6003TJKZ).
_LLAMA_DEATH_LABEL = "llama_server"

logger = logging.getLogger("llama-proxy.wedge")


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    from ..backend_health import _srv as _shared_srv
    return _shared_srv()


# ===================================================================
# GPU-wedge detection (LP-0MS91DHQ9003EGB0, LP-0MW5K8ZW3005SEA8)
#
# A wedged llama-server is still HTTP-reachable but its GPU kernels never
# complete: the GPU draws compute-level power while every slot is idle and no
# tokens are produced. The existing watchdog/health probes only catch dead or
# unreachable workers, so the wedge was invisible to self-healing. Detection
# is proxy-side only (no llama.cpp changes):
#
#   signature = compute-level power (>= llama_gpu_wedge_power_low_threshold_watts)
#               AND all slots idle
#
# sustained for ``llama_gpu_wedge_idle_checks_required`` consecutive health
# loop iterations. GPU power draw is read from sysfs (AMD hwmon
# power1_average, uW) with a rocm-smi fallback; when no power source is
# available the probe fails closed and returns no verdict - a missing power
# source is never a wedge.
#
# History (LP-0MW5K8ZW3005SEA8): detection used to key on the
# GPU busy counter plus the power discriminator. On the Strix Halo
# APU (Radeon 8060S iGPU, gfx1151) that counter is a two-state detector, not
# a percentage: measured 2026-11-06 on kernel 7.0.0-38 (LP-0MW5K7WKZ005VJXA)
# it reads 99.2% idle-with-display against 98.7% when computing - it inverts
# across the band that matters, so the busy+idle signature fired on every
# idle period and caused constant unload/reload churn (144 unloads in one
# day, 2026-07-31, LP-0MS9GA6QL007XTBP). The counter is therefore not read
# at all here; the power discriminator plus slot idleness express the same
# signature without it, and the proxy's no-progress watchdog
# (local_dispatch_no_progress_timeout_seconds, LP-0MUCEFB8E003YVFF) covers
# the "no token progress" half. A slotted engine stuck at idle power
# (LP-0MS91D782006XIR6 drew only ~53 W) is indistinguishable from normal idle
# from the power sensor alone and is never treated as a wedge - conservative
# by design, since enabling detection is opt-in.
#
# Detection is disabled by default (llama_gpu_wedge_detection_enabled:
# false) because the idle-with-display band (~33-54 W, p50 45 W) sits close
# to transient compute (>= p75 92 W) on this APU and the sensor alone cannot
# separate "idle" from "stalled at idle power". This item removed the
# counter dependency; it does not enable self-healing.
# ===================================================================


# ===================================================================
# GPU power draw sources
# ===================================================================


def _gpu_power_draw_sysfs(base: str = "/sys/class/drm/card0/device") -> float | None:
    """Read GPU power draw in watts from sysfs hwmon power1_average (uW).

    Scans the hwmon nodes under the card0 device dir and returns the first
    ``power1_average`` found (e.g. amdgpu PPT on the Strix Halo APU). None
    when unavailable.
    """
    try:
        hwmon_root = Path(base) / "hwmon"
        for hwmon_dir in sorted(hwmon_root.glob("hwmon*")):
            power_file = hwmon_dir / "power1_average"
            if power_file.exists():
                val = float(power_file.read_text().strip()) / 1_000_000.0
                if val >= 0:
                    return val
    except Exception:
        return None
    return None


def _gpu_power_draw_rocm_smi() -> float | None:
    """Fallback: parse ``rocm-smi --showpower`` output (W). None when unavailable."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpower"],
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
        for line in out.splitlines():
            if "Power" not in line:
                continue
            val_str = line.split(":")[-1].strip().rstrip("W").strip()
            try:
                val = float(val_str)
            except ValueError:
                continue
            return val
    except Exception:
        return None
    return None


def _read_gpu_power_draw() -> float | None:
    """Best-effort GPU power draw (W): sysfs first, then rocm-smi."""
    val = _gpu_power_draw_sysfs()
    if val is None:
        val = _gpu_power_draw_rocm_smi()
    return val


def _slots_all_idle(slots_payload) -> bool:
    """True when every slot in a /slots payload reports is_processing == False."""
    if not isinstance(slots_payload, list):
        return False
    if not slots_payload:
        return True
    return all(
        not bool(s.get("is_processing"))
        for s in slots_payload
        if isinstance(s, dict)
    )


# GPU state classification from the wedge probe.
WEDGE_STATE_NONE = "none"  # no wedge signature
WEDGE_STATE_WEDGE = "wedge"  # compute-level power + all slots idle: wedged


def _gpu_wedge_discriminate(
    power_watts: float | None,
    slots_idle: bool,
    power_low_threshold: float,
) -> str:
    """Classify the GPU state from the wedge probe.

    Returns one of the ``WEDGE_STATE_*`` constants (LP-0MW5K8ZW3005SEA8):

    - ``none``: not a wedge signature (power below the compute threshold, a
      slot is processing, or power is unavailable — fail closed, never a
      wedge from missing data).
    - ``wedge``: the GPU is drawing compute-level power
      (>= ``power_low_threshold`` W) while every slot is idle — GPU kernels
      active with no work to show for them.
    """
    if power_watts is None or not slots_idle or power_watts < power_low_threshold:
        return WEDGE_STATE_NONE
    return WEDGE_STATE_WEDGE


#: Consecutive /slots query failures seen by the wedge probe (LP-0MW5K8ZW3005SEA8
#: AC6). The probe logs each failure with the running streak so a slot-query
#: outage is visible even though the probe fails open — a failed slot query
#: never flips ``slots_idle`` True and never produces a wedge verdict.
_slot_query_failure_streak = 0


async def _probe_gpu_wedge(
    host: str,
    port: int,
    timeout: float = 5.0,
    model: str | None = None,
) -> tuple[float | None, bool]:
    """Return ``(gpu_power_watts, all_slots_idle)`` for a model instance.

    *model* is the router model id the probed instance serves and is sent as
    ``?model=``: llama-server requires it and its router answers a model-less
    ``/slots`` with HTTP 400 (LP-0MVP7HV7Q001AJF8 AC2). When no model is known
    the slot query is skipped and ``all_slots_idle`` stays False — fail-open,
    never an exception, and never a wedge verdict derived from missing data.
    Each slot-query failure is logged with its running streak (AC6: no silent
    blindness to a slot-query outage).
    """
    global _slot_query_failure_streak
    power = _read_gpu_power_draw()
    slots_idle = False
    if not model:
        return power, slots_idle
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        try:
            resp = await client.get(
                f"http://{host}:{port}/slots?model={model}", timeout=timeout
            )
            if resp.status_code == 200:
                slots_idle = _slots_all_idle(resp.json())
        finally:
            await client.aclose()
    except Exception as exc:
        _slot_query_failure_streak += 1
        logger.warning(
            "gpu_wedge probe: /slots query failed for model %s at %s:%s "
            "(%d consecutive failure(s)): %s: %s",
            model,
            host,
            port,
            _slot_query_failure_streak,
            type(exc).__name__,
            exc,
        )
    else:
        if _slot_query_failure_streak:
            _slot_query_failure_streak = 0
    return power, slots_idle


# ===================================================================
# Router self-healing
# ===================================================================


def _proc_handle_alive(proc) -> bool:
    """True when *proc* is a live process handle (``poll()`` returns None)."""
    if proc is None:
        return False
    poll = getattr(proc, "poll", None)
    if not callable(poll):
        return False
    try:
        return poll() is None
    except Exception:
        return False


def _adopt_reachable_backend(srv, llama_port: int) -> bool:
    """Adopt a reachable backend this proxy holds no process handle for.

    LP-0MVZT9L3N001XABB AC2: the watchdog used to only reset
    ``backend_ready`` when the port answered with ``llama_process is None``,
    which left ``/health``'s process-handle based ``llama_server_running``
    (and therefore ``/llama/local/status`` slot reporting) stuck on the
    degraded verdict for as long as the orphan served the port. Installing
    the adopted-port sentinel makes the backend a first-class managed server
    again: liveness is reported, ``stop_llama_server`` can kill it by port, and
    the sentinel's ``pid`` of 0 keeps ``_worker_process_unhealthy`` inert.

    Returns True when a sentinel was installed.
    """
    if _proc_handle_alive(getattr(srv, "llama_process", None)):
        return False
    try:
        from ..lifecycle import _AdoptedLlamaProcess

        srv.llama_process = _AdoptedLlamaProcess(port=int(llama_port))
    except Exception:
        return False
    try:
        srv.logger.warning(
            "watchdog: adopted the reachable but unmanaged llama-server on port %s "
            "(no process handle); /health and slot reporting now reflect it",
            llama_port,
        )
    except Exception:
        pass
    return True


async def _settle_unready_self_heal_spawn(srv, proc, llama_port: int) -> bool:
    """Resolve a self-heal spawn whose readiness wait expired (AC1).

    LP-0MVZT9L3N001XABB: on 2026-11-02 21:39Z a host-first router spawn that
    was still loading its presets when ``wait_for_llama_server``'s budget ran
    out was declared failed and left running but unadopted. It kept port 8080,
    so the next attempt died on ``couldn't bind HTTP server socket`` and
    self-heal exhausted — while the orphan served the local route fine.

    A spawn is therefore never abandoned:

    * **reachable** — adopt it (the live ``Popen`` when the handle survived,
      else an adopted-port sentinel for a surviving descendant) and report
      success, so the work already done is not thrown away;
    * **unreachable** — terminate its process group so the port is released
      for the next attempt instead of blocking it.

    Returns True when the spawn was adopted.
    """
    try:
        reachable = bool(await srv._probe_backend_reachable(llama_port))
    except Exception:
        reachable = False

    if reachable:
        live_handle = _proc_handle_alive(proc)
        if live_handle:
            srv.llama_process = proc
        else:
            _adopt_reachable_backend(srv, llama_port)
        srv.backend_ready = True
        srv.backend_recovery_state["last_failure"] = None
        try:
            srv.logger.warning(
                "self-heal: backend on port %s became reachable after the readiness "
                "wait expired; adopting the spawn (pid=%s, live_handle=%s) instead of "
                "leaving it unadopted",
                llama_port,
                getattr(proc, "pid", "N/A"),
                live_handle,
            )
        except Exception:
            pass
        return True

    # Unreachable: the spawn must not be left holding the port.
    try:
        kill = getattr(srv, "_kill_process_group", None)
        if callable(kill):
            kill(proc, srv.logger)
        else:  # pragma: no cover - defensive fallback for partial servers
            terminate = getattr(proc, "terminate", None)
            if callable(terminate):
                terminate()
        # The handle no longer represents a live backend: clear it so the
        # failure path cannot re-settle (and re-kill) the same spawn.
        if getattr(srv, "llama_process", None) is proc:
            srv.llama_process = None
        srv.logger.warning(
            "self-heal: terminated the unadopted spawn (pid=%s) that never became "
            "reachable on port %s, freeing the port for the next attempt",
            getattr(proc, "pid", "N/A"),
            llama_port,
        )
    except Exception:
        try:
            srv.logger.exception(
                "self-heal: failed to terminate the unadopted spawn on port %s",
                llama_port,
            )
        except Exception:
            pass
    return False


async def _port_held_by_unmanaged_backend(srv, llama_port: int) -> bool:
    """True when the port answers but this proxy owns no live process for it.

    LP-0MVZVZLCS000LQNB AC4: the respawn guard. A backend that answers on the
    configured port while ``llama_process`` is absent (or dead) is already
    serving the local route — spawning a second one cannot bind and would
    produce the restart cascade this item exists to stop.
    """
    if _proc_handle_alive(getattr(srv, "llama_process", None)):
        return False
    try:
        return bool(await srv._probe_backend_reachable(llama_port))
    except Exception:
        return False


async def _settle_leftover_spawn(srv, llama_port: int) -> bool:
    """Settle a spawn left alive by a failed self-heal attempt (AC1).

    ``_settle_unready_self_heal_spawn`` handles the ordinary "readiness wait
    returned False" path. This wrapper covers the exceptional path (the
    readiness wait raised, or the attempt failed after the handle was
    installed): a live leftover spawn is adopted when reachable and terminated
    when not, so no failure mode leaves it unadopted holding the port.
    """
    leftover = getattr(srv, "llama_process", None)
    if not _proc_handle_alive(leftover):
        return False
    return await _settle_unready_self_heal_spawn(srv, leftover, llama_port)


async def _attempt_router_self_heal(reason: str = "unspecified") -> bool:
    """Attempt router-mode self-healing with capped exponential backoff.

    *reason* is the death/cause that triggered this restart attempt. It is
    logged on every attempt line as ``restart_reason=<reason>`` and recorded in
    ``backend_recovery_state["last_restart_reason"]`` so a restart is never
    anonymous (LP-0MVZVZLCS000LQNB AC2), and it is attached to the
    ``backend_restart`` evictions of the prefills the restart abandons (AC3).
    """
    srv = _srv()

    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    max_attempts = max(
        1, int(server_cfg.get("llama_self_heal_max_attempts", 3) or 3)
    )
    window_seconds = max(
        1, int(server_cfg.get("llama_self_heal_window_seconds", 300) or 300)
    )
    base_backoff = max(
        0.0,
        float(server_cfg.get("llama_self_heal_backoff_base_seconds", 1.0) or 1.0),
    )
    startup_timeout = int(server_cfg.get("llama_startup_timeout", 300) or 300)
    llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
    retry_after = srv._self_heal_retry_after_seconds()

    now_ts = time.time()
    attempts = srv.backend_recovery_state.get("attempt_timestamps", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts = srv._prune_recovery_attempts(attempts, now_ts, window_seconds)

    srv.backend_recovery_state["attempt_timestamps"] = attempts
    srv.backend_recovery_state["max_attempts"] = max_attempts
    srv.backend_recovery_state["window_seconds"] = window_seconds
    srv.backend_recovery_state["retry_after_seconds"] = retry_after

    if len(attempts) >= max_attempts:
        srv.backend_recovery_state["in_progress"] = False
        srv.backend_recovery_state["last_failure"] = (
            f"self-heal throttled: max {max_attempts} attempts in "
            f"{window_seconds}s"
        )
        srv.logger.error(
            "self-heal giving up: max attempts reached (%s attempts in %ss); "
            "manual intervention required",
            max_attempts,
            window_seconds,
        )
        return False

    srv.backend_recovery_state["in_progress"] = True
    srv.backend_recovery_state["last_restart_reason"] = reason
    remaining = max_attempts - len(attempts)

    try:
        for local_attempt in range(remaining):
            attempt_started = time.time()
            attempts.append(attempt_started)
            srv.backend_recovery_state["attempt_timestamps"] = attempts
            attempt_number = len(attempts)

            srv.logger.warning(
                "self-heal attempt %s/%s started (window=%ss, restart_reason=%s)",
                attempt_number,
                max_attempts,
                window_seconds,
                reason,
            )

            try:
                # LP-0MVZVZLCS000LQNB AC4: never spawn over a backend that is
                # already answering on the port. A crashed backend that a
                # previous (or external) spawn has already replaced is adopted
                # instead — spawning here is exactly the 21:40:16Z cascade
                # where a second router died on
                # "couldn't bind HTTP server socket ... port: 8080".
                #
                # Only retries are gated: the first attempt is the watchdog's
                # deliberate recovery (it has already established that the
                # backend is unreachable), so a router that answers /models
                # while its child is dead still gets its restart.
                if local_attempt > 0 and await _port_held_by_unmanaged_backend(
                    srv, llama_port
                ):
                    adopted = _adopt_reachable_backend(srv, llama_port)
                    srv.backend_ready = True
                    srv.backend_recovery_state["last_failure"] = None
                    srv.logger.warning(
                        "self-heal: not respawning — port %s is already served by "
                        "an unmanaged backend (restart_reason=%s, adopted=%s); no "
                        "new process started",
                        llama_port,
                        "port_held_by_unmanaged_backend",
                        adopted,
                    )
                    return True

                # LP-0MVZVZLCS000LQNB AC3: a restart abandons the prefills in
                # flight; record them so the churn is visible in the eviction
                # accounting instead of looking like a capacity problem.
                try:
                    from .. import admission_events as _admission_events

                    _admission_events.record_backend_restart_evictions(
                        srv, restart_reason=reason
                    )
                except Exception:
                    pass

                restarted = srv.start_llama_server(None)
                if restarted is None:
                    raise RuntimeError("start_llama_server returned None")

                srv.llama_process = restarted
                srv.backend_ready = await srv.wait_for_llama_server(startup_timeout)
                if srv.backend_ready:
                    srv.backend_recovery_state["last_failure"] = None
                    srv.logger.info(
                        "self-heal succeeded on attempt %s/%s",
                        attempt_number,
                        max_attempts,
                    )
                    return True

                # LP-0MVZT9L3N001XABB AC1: a spawn that outlived the readiness
                # wait is adopted when it is reachable and terminated when it
                # is not — it is never left unadopted holding the port.
                if await _settle_unready_self_heal_spawn(srv, restarted, llama_port):
                    srv.logger.info(
                        "self-heal succeeded on attempt %s/%s (adopted a "
                        "late-ready backend)",
                        attempt_number,
                        max_attempts,
                    )
                    return True

                raise RuntimeError("wait_for_llama_server returned False")
            except Exception as exc:
                # LP-0MVZT9L3N001XABB AC1: a spawn that is still alive when an
                # attempt fails must not be abandoned holding the port.
                try:
                    if await _settle_leftover_spawn(srv, llama_port):
                        srv.logger.info(
                            "self-heal recovered via a leftover spawn on "
                            "attempt %s/%s",
                            attempt_number,
                            max_attempts,
                        )
                        return True
                except Exception:
                    pass
                srv.backend_ready = False
                srv.llama_process = None
                srv.current_model = None
                srv.backend_recovery_state["last_failure"] = str(exc)
                srv.logger.error(
                    "self-heal attempt %s/%s failed: %s",
                    attempt_number,
                    max_attempts,
                    exc,
                )

            if local_attempt < remaining - 1:
                delay = base_backoff * (2**local_attempt)
                srv.logger.warning(
                    "self-heal backoff sleeping %.1fs before retry", delay
                )
                await asyncio.sleep(delay)

        srv.logger.error(
            "self-heal exhausted after %s attempt(s) within %ss; "
            "manual intervention required",
            remaining,
            window_seconds,
        )
        return False
    finally:
        srv.backend_recovery_state["in_progress"] = False


# ===================================================================
# Backend watchdog
# ===================================================================

async def _backend_watchdog_loop() -> None:
    """Watch local backend process and trigger best-effort recovery.

    In router mode, when ``backend_ready`` is False but the backend
    (llama-server) is actually reachable on its port, this function
    resets ``backend_ready`` to True without requiring a full process
    restart. This handles the case where ``stop_llama_server`` or a
    transient failure sets ``backend_ready=False`` while the independent
    host llama-server remains healthy (LP-0MRCQW0HC000J4F9).
    """
    srv = _srv()

    while True:
        try:
            interval = float(
                srv.config.get("server", {}).get(
                    "llama_watchdog_interval_seconds", 5.0
                )
                or 5.0
            )
            await asyncio.sleep(max(0.0, interval))

            proc = srv.llama_process

            # LP-0MRCQW0HC000J4F9: Probe backend before attempting restart.
            # In router mode the llama-server may be running independently
            # on the host, not as a proxy-managed process. When the process
            # is None or has exited, first check if the backend is actually
            # reachable before attempting a full restart.
            router_mode = bool(
                srv.config.get("server", {}).get("llama_router_mode", False)
            )
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
            llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)

            # LP-0MW3K7XJ30027RV6: keep the OOM ranking on the *backend* — the
            # router and the model children it forks — not only on the
            # start-script wrapper the proxy holds a handle for. Throttled
            # internally, best-effort, and it records a documented gap when the
            # backend cannot be found or scored.
            try:
                from ..lifecycle import _reapply_backend_oom_protection

                _reapply_backend_oom_protection()
            except Exception:
                pass

            # LP-0MW3K7YO1005C41X: re-assert process-group isolation between
            # the GPU backends (throttled internally, best-effort). A violation
            # means a fault in this backend could now signal the TTS tree — log
            # it loudly instead of letting the cascade happen silently.
            try:
                from ..backend_isolation import assert_backend_isolation

                assert_backend_isolation(srv, srv.logger)
            except Exception:
                pass

            # LP-0MW3MOMD9007DC26 AC1/AC3: an adopted sentinel (pid=0, its
            # poll() always returns None) can never trip the poll()-based death
            # checks below — its only liveness signal is the port itself. Probe
            # reachability every interval and, after ``llama_adopted_unreachable_probes``
            # consecutive unreachable probes, declare the death: log a death
            # line (the true signal when the handle carries a decodable pid,
            # else the adopted-unreachable class), record it, clear the handle
            # and run the normal router self-heal.
            if getattr(proc, "adopted", False) is True:
                reachable = await srv._probe_backend_reachable(llama_port)
                if reachable:
                    # Still alive: make backend_ready reflect reality (a
                    # previous failure may have left it False), and reset the
                    # unreachable-probe streak so a later death needs the full
                    # window again (a transient dip is not a partial death).
                    srv.backend_recovery_state["adopted_unreachable_streak"] = 0
                    if not srv.backend_ready:
                        srv.backend_ready = True
                        srv.backend_recovery_state["last_failure"] = None
                        try:
                            srv.logger.info(
                                "watchdog: adopted llama-server reachable on port %d, "
                                "resetting backend_ready to True",
                                llama_port,
                            )
                        except Exception:
                            pass
                    continue

                # Unreachable: accumulate a streak so a single transient probe
                # failure does not restart a live backend.
                streak = int(
                    srv.backend_recovery_state.get("adopted_unreachable_streak", 0)
                    or 0
                ) + 1
                srv.backend_recovery_state["adopted_unreachable_streak"] = streak
                max_probes = max(
                    1,
                    int(
                        server_cfg.get("llama_adopted_unreachable_probes", 3) or 3
                    ),
                )
                if streak < max_probes:
                    try:
                        srv.logger.warning(
                            "watchdog: adopted llama-server unreachable on port %d "
                            "(probe %d/%d)",
                            llama_port,
                            streak,
                            max_probes,
                        )
                    except Exception:
                        pass
                    continue

                # Declared dead after N consecutive unreachable probes.
                srv.backend_ready = False
                srv._record_backend_signal("other_failures")
                death = _death_capture.probe_death(proc)
                if death is None:
                    reason = "llama_server_adopted_unreachable"
                    describe = f"unreachable on port {llama_port} (pid=0, no wait status)"
                else:
                    reason = death.restart_reason(_LLAMA_DEATH_LABEL)
                    describe = death.describe()
                try:
                    srv.logger.error(
                        "watchdog detected adopted llama-server death: %s "
                        "(restart_reason=%s) model=%s",
                        describe,
                        reason,
                        srv.current_model,
                    )
                except Exception:
                    pass
                _death_capture.record_death(
                    srv,
                    _LLAMA_DEATH_LABEL,
                    death,
                    extra={"restart_reason": reason, "adopted": True},
                    state_attr="backend_recovery_state",
                )
                srv.backend_recovery_state["last_restart_reason"] = reason
                srv.backend_recovery_state["last_failure"] = (
                    f"adopted llama-server unreachable on port {llama_port}"
                )
                srv.llama_process = None
                srv.current_model = None
                if router_mode:
                    recovered = await srv._attempt_router_self_heal(reason=reason)
                    try:
                        srv.logger.info(
                            "watchdog adopted-backend self-heal recovered=%s",
                            recovered,
                        )
                    except Exception:
                        pass
                continue

            if proc is None:
                if router_mode and not srv.backend_ready:
                    # Probe the backend before attempting restart
                    reachable = await srv._probe_backend_reachable(llama_port)
                    if reachable:
                        # LP-0MVZT9L3N001XABB AC2: adopt the reachable backend
                        # so /health and slot reporting stop reporting the
                        # sticky degraded verdict, not just backend_ready.
                        adopted = _adopt_reachable_backend(srv, llama_port)
                        srv.logger.info(
                            "watchdog: llama_process is None but backend is reachable on port %d, "
                            "resetting backend_ready to True (adopted=%s)",
                            llama_port,
                            adopted,
                        )
                        srv.backend_ready = True
                        srv.backend_recovery_state["last_failure"] = None
                        continue

                    srv.logger.warning(
                        "watchdog: llama_process is None and backend unreachable, "
                        "attempting restart"
                    )
                    recovered = await srv._attempt_router_self_heal(
                        reason=f"backend_unreachable(port={llama_port})"
                    )
                    srv.logger.info(
                        "watchdog restart-from-none recovered=%s", recovered
                    )
                continue

            death = _death_capture.probe_death(proc)

            worker_unhealthy = False
            if death is None:
                worker_unhealthy = srv._worker_process_unhealthy(proc)
                if not worker_unhealthy:
                    # LP-0MRCQW0HC000J4F9: Worker is healthy but backend_ready may
                    # still be False from a prior failure. Reset if backend reachable.
                    if router_mode and not srv.backend_ready:
                        reachable = await srv._probe_backend_reachable(llama_port)
                        if reachable:
                            srv.logger.info(
                                "watchdog: worker healthy but backend_ready=False, "
                                "backend reachable on port %d, resetting backend_ready to True",
                                llama_port,
                            )
                            srv.backend_ready = True
                            srv.backend_recovery_state["last_failure"] = None
                    continue

            if death is None:
                srv.logger.error(
                    "watchdog detected unhealthy worker while main process "
                    "is alive model=%s",
                    srv.current_model,
                )
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                except Exception:
                    pass
            else:
                # LP-0MW3K7WD6003TJKZ AC1/AC2: report the true termination
                # cause (signal name and 128+N code) instead of a blanket
                # ``exit code=137``, and keep the second-resolution death
                # window as the attribution evidence.
                srv.logger.error(
                    "watchdog detected llama-server death: %s "
                    "(restart_reason=%s) model=%s",
                    death.describe(),
                    death.restart_reason(_LLAMA_DEATH_LABEL),
                    srv.current_model,
                )
                _death_capture.record_death(
                    srv,
                    _LLAMA_DEATH_LABEL,
                    death,
                    extra={"model": srv.current_model},
                    state_attr="backend_recovery_state",
                )

            # LP-0MRCQW0HC000J4F9: Before marking backend_ready=False and
            # triggering full restart, probe whether the backend is still
            # reachable (independent host llama-server in router mode).
            if router_mode:
                reachable = await srv._probe_backend_reachable(llama_port)
                if reachable:
                    # LP-0MVZT9L3N001XABB AC2: adopt instead of only clearing
                    # the handle, so /health and slot reporting agree with the
                    # working route. The exited handle is cleared first: the
                    # adoption installs the port sentinel that replaces it.
                    srv.llama_process = None
                    srv.current_model = None
                    adopted = _adopt_reachable_backend(srv, llama_port)
                    srv.logger.info(
                        "watchdog: process exited but backend reachable on port %d, "
                        "resetting backend_ready to True (process exit: %s, "
                        "adopted=%s)",
                        llama_port,
                        death.describe() if death is not None else "worker_unhealthy",
                        adopted,
                    )
                    srv.backend_ready = True
                    srv.backend_recovery_state["last_failure"] = None
                    continue

            srv.backend_ready = False
            srv._record_backend_signal("other_failures")
            srv.llama_process = None
            srv.current_model = None

            if router_mode:
                recovered = await srv._attempt_router_self_heal(
                    reason=(
                        "worker_unhealthy"
                        if death is None
                        else death.restart_reason(_LLAMA_DEATH_LABEL)
                    )
                )
                srv.logger.info(
                    "watchdog router self-heal recovered=%s", recovered
                )

        except asyncio.CancelledError:
            return
        except Exception:
            srv.logger.exception("watchdog loop error")


# ===================================================================
# Router model health monitoring
# ===================================================================

async def _unload_and_reload_model(
    srv,
    router_host: str,
    router_port: int,
    model_id: str,
    reason: str,
) -> bool:
    """Unload and reload a model instance (shared by unreachable and GPU-wedge recovery)."""
    srv.logger.info("model_health: unloading model %s (%s)", model_id, reason)
    try:
        client = (
            srv._http_client if srv._http_client else httpx.AsyncClient(timeout=10.0)
        )
        try:
            await client.post(
                f"http://{router_host}:{router_port}/models/unload",
                json={"model": model_id},
                timeout=10.0,
            )
        finally:
            if not srv._http_client:
                await client.aclose()
    except Exception as exc:
        srv.logger.warning(
            "model_health: unload request for %s failed: %s",
            model_id,
            exc,
        )

    srv.logger.info("model_health: reloading model %s", model_id)
    loaded = await srv.router_load_model(model_id)
    if loaded:
        srv.logger.info(
            "model_health: successfully reloaded model %s",
            model_id,
        )
    else:
        srv.logger.error(
            "model_health: failed to reload model %s",
            model_id,
        )
    return bool(loaded)


async def _router_model_health_loop() -> None:
    """Periodically check loaded models' reachability in router mode.

    Guardrails to reduce false positives:
    - legacy interval-key fallback (llama_health_check_interval)
    - initial grace window after model (re)load or port change
    - multi-attempt probing before counting a failure
    - consecutive failure threshold before unload/reload
    """
    from ..backend_health import (
        _coerce_float,
        _coerce_int,
        _extract_model_port_from_args,
        _is_self_healing_active,
        _probe_model_instance_with_retries,
    )

    srv = _srv()

    # Stateful counters across loop iterations
    consecutive_failures: dict[str, int] = {}
    observed_ports: dict[str, int] = {}
    port_first_seen_at: dict[str, float] = {}
    wedge_checks: dict[str, int] = {}
    gpu_wedge_source_logged = False

    while True:
        try:
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

            interval_config = server_cfg.get(
                "llama_model_health_interval_seconds",
                server_cfg.get("llama_health_check_interval", 30.0),
            )
            interval = _coerce_float(interval_config, 30.0)

            failures_before_recovery = max(
                1,
                _coerce_int(
                    server_cfg.get("llama_model_health_failures_before_recovery", 2),
                    2,
                ),
            )

            probe_timeout = max(
                0.5,
                _coerce_float(
                    server_cfg.get("llama_model_health_probe_timeout_seconds", 5.0),
                    5.0,
                ),
            )

            probe_attempts = max(
                1,
                _coerce_int(
                    server_cfg.get("llama_model_health_probe_attempts", 2),
                    2,
                ),
            )

            probe_backoff = max(
                0.0,
                _coerce_float(
                    server_cfg.get("llama_model_health_probe_backoff_seconds", 0.5),
                    0.5,
                ),
            )

            grace_period_seconds = max(
                0.0,
                _coerce_float(
                    server_cfg.get("llama_model_health_grace_period_seconds", 15.0),
                    15.0,
                ),
            )

            # GPU-wedge detection configuration (LP-0MS91DHQ9003EGB0 /
            # LP-0MW5K8ZW3005SEA8). Disabled by default: on the Strix Halo APU
            # the idle-with-display band (~33-54 W, p50 45 W) sits close to
            # transient compute (>= p75 92 W), and a wedged engine
            # (LP-0MS91D782006XIR6, ~53 W idle power) is indistinguishable from
            # normal idle from the power sensor alone. The old busy-counter
            # read (removed here) false-positived on every idle period; the
            # power-only signature below keeps that false-positive profile but
            # drops the unusable read (LP-0MW5K7WKZ005VJXA, monitoring/README.md
            # "Why not amd-smi, rocm-smi or amdtop?").
            gpu_wedge_enabled = bool(
                server_cfg.get("llama_gpu_wedge_detection_enabled", False)
            )
            gpu_wedge_idle_checks = max(
                1,
                _coerce_int(
                    server_cfg.get("llama_gpu_wedge_idle_checks_required", 2),
                    2,
                ),
            )
            # Power below this (W) means the GPU is not genuinely computing:
            # idle slots at below-threshold power are NOT a wedge, and a
            # missing power source yields no verdict at all (fail closed).
            gpu_wedge_power_low_threshold = max(
                0.0,
                _coerce_float(
                    server_cfg.get("llama_gpu_wedge_power_low_threshold_watts", 90.0),
                    90.0,
                ),
            )

            await asyncio.sleep(max(5.0, interval))

            router_mode = bool(server_cfg.get("llama_router_mode", False))
            if not router_mode:
                continue

            # Don't interfere while the watchdog is actively recovering
            if _is_self_healing_active():
                continue

            models_data = await srv.router_list_models()
            if not isinstance(models_data, dict):
                continue

            models_payload = models_data.get("data") or models_data.get("models") or []
            if not isinstance(models_payload, list):
                continue

            router_host = "127.0.0.1"
            try:
                router_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
            except Exception:
                router_port = 8080

            now_ts = time.time()
            loaded_model_ids: set[str] = set()

            for model_entry in models_payload:
                if not isinstance(model_entry, dict):
                    continue

                model_id = model_entry.get("id")
                raw_status = model_entry.get("status", {})

                if isinstance(raw_status, str):
                    status_value = raw_status.lower()
                    args = []
                elif isinstance(raw_status, dict):
                    status_value = str(raw_status.get("value", "")).lower()
                    args = raw_status.get("args", [])
                else:
                    continue

                if status_value != "loaded" or not model_id:
                    continue

                loaded_model_ids.add(model_id)

                port = _extract_model_port_from_args(args)
                if port is None or port <= 0:
                    srv.logger.debug(
                        "model_health: cannot determine port for loaded model %s, skipping",
                        model_id,
                    )
                    continue

                # Reset tracking when the loaded instance port changes.
                prior_port = observed_ports.get(model_id)
                if prior_port != port:
                    observed_ports[model_id] = port
                    port_first_seen_at[model_id] = now_ts
                    consecutive_failures[model_id] = 0

                first_seen = port_first_seen_at.get(model_id, now_ts)
                age_seconds = max(0.0, now_ts - first_seen)
                if grace_period_seconds > 0 and age_seconds < grace_period_seconds:
                    srv.logger.debug(
                        "model_health: skipping probe for %s (port %d) during grace window %.1fs/%.1fs",
                        model_id,
                        port,
                        age_seconds,
                        grace_period_seconds,
                    )
                    continue

                reachable = await _probe_model_instance_with_retries(
                    router_host,
                    port,
                    timeout=probe_timeout,
                    attempts=probe_attempts,
                    backoff_seconds=probe_backoff,
                )
                if reachable:
                    if consecutive_failures.get(model_id, 0) > 0:
                        srv.logger.info(
                            "model_health: model %s recovered after %d failed probe(s)",
                            model_id,
                            consecutive_failures.get(model_id, 0),
                        )
                    consecutive_failures[model_id] = 0

                    # GPU-wedge detection: the instance is reachable but the GPU
                    # may be wedged (compute-level power with idle slots, no
                    # progress). Skip embeddings instances; a missing power source
                    # yields no verdict (fail closed, never a wedge from missing
                    # data).
                    if (
                        gpu_wedge_enabled
                        and "--embeddings" not in args
                    ):
                        power_watts, slots_idle = await _probe_gpu_wedge(
                            router_host,
                            port,
                            timeout=probe_timeout,
                            model=model_id,
                        )
                        if power_watts is None:
                            if not gpu_wedge_source_logged:
                                srv.logger.warning(
                                    "model_health: GPU power source unavailable "
                                    "(no sysfs power1_average, no rocm-smi); "
                                    "GPU-wedge detection skipped for this process"
                                )
                                gpu_wedge_source_logged = True
                            continue
                        state = _gpu_wedge_discriminate(
                            power_watts,
                            slots_idle,
                            gpu_wedge_power_low_threshold,
                        )
                        if state == WEDGE_STATE_NONE:
                            wedge_checks[model_id] = 0
                            if getattr(srv, "gpu_wedge_detected", False):
                                srv.logger.info(
                                    "model_health: GPU wedge cleared for model %s "
                                    "(power=%.0fW, slots_idle=%s)",
                                    model_id,
                                    power_watts,
                                    slots_idle,
                                )
                                srv.gpu_wedge_detected = False
                                srv.gpu_wedge_detected_at = None
                                srv.gpu_wedge_last_model = None
                        else:  # WEDGE_STATE_WEDGE
                            count = wedge_checks.get(model_id, 0) + 1
                            wedge_checks[model_id] = count
                            if count >= gpu_wedge_idle_checks:
                                srv.logger.error(
                                    "model_health: GPU wedge detected for model %s "
                                    "(port %d): power %.0fW (compute-level) with all "
                                    "slots idle for %d consecutive check(s); "
                                    "triggering recovery",
                                    model_id,
                                    port,
                                    power_watts,
                                    count,
                                )
                                srv._record_backend_signal("gpu_wedge")
                                srv.gpu_wedge_detected = True
                                srv.gpu_wedge_detected_at = time.time()
                                srv.gpu_wedge_last_model = model_id
                                await _unload_and_reload_model(
                                    srv,
                                    router_host,
                                    router_port,
                                    model_id,
                                    reason=f"gpu_wedge (power {power_watts:.0f}W)",
                                )
                                wedge_checks[model_id] = 0
                                port_first_seen_at[model_id] = time.time()
                            else:
                                srv.logger.warning(
                                    "model_health: possible GPU wedge for model %s "
                                    "(port %d): power %.0fW with all slots idle "
                                    "(%d/%d checks)",
                                    model_id,
                                    port,
                                    power_watts,
                                    count,
                                    gpu_wedge_idle_checks,
                                )
                    continue

                failure_count = consecutive_failures.get(model_id, 0) + 1
                consecutive_failures[model_id] = failure_count

                if failure_count < failures_before_recovery:
                    srv.logger.warning(
                        "model_health: model %s (port %d) probe failed (%d/%d); delaying recovery",
                        model_id,
                        port,
                        failure_count,
                        failures_before_recovery,
                    )
                    continue

                srv.logger.error(
                    "model_health: model %s (port %d) is loaded but unreachable for %d consecutive probe cycle(s), triggering recovery",
                    model_id,
                    port,
                    failure_count,
                )

                await _unload_and_reload_model(
                    srv,
                    router_host,
                    router_port,
                    model_id,
                    reason="unreachable",
                )

                # Reset counters and re-apply grace period after recovery attempt
                consecutive_failures[model_id] = 0
                port_first_seen_at[model_id] = time.time()

            # Prune state for models no longer loaded
            stale_ids = [model_id for model_id in list(consecutive_failures.keys()) if model_id not in loaded_model_ids]
            for stale_id in stale_ids:
                consecutive_failures.pop(stale_id, None)
                observed_ports.pop(stale_id, None)
                port_first_seen_at.pop(stale_id, None)
                wedge_checks.pop(stale_id, None)

        except asyncio.CancelledError:
            return
        except Exception:
            srv.logger.exception("model health loop error")


async def _probe_model_instance_with_retries(
    host: str,
    port: int,
    timeout: float = 5.0,
    attempts: int = 2,
    backoff_seconds: float = 0.5,
) -> bool:
    """Probe a model instance with retries to reduce transient false negatives."""
    tries = max(1, int(attempts or 1))
    pause = max(0.0, float(backoff_seconds or 0.0))

    for attempt_idx in range(tries):
        reachable = await _probe_model_instance(host, port, timeout=timeout)
        if reachable:
            return True
        if attempt_idx < tries - 1 and pause > 0:
            await asyncio.sleep(pause)
    return False


async def _probe_model_instance(
    host: str, port: int, timeout: float = 5.0
) -> bool:
    """Probe whether a model instance is reachable on its port.

    Performs a simple GET to ``/health`` on the given host:port.
    Returns True if the endpoint responds with HTTP 200.
    """
    if port <= 0:
        return False
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        try:
            url = f"http://{host}:{port}/health"
            response = await client.get(url, timeout=timeout)
            return response.status_code == 200
        finally:
            await client.aclose()
    except Exception:
        return False
