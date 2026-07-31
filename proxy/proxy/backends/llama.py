"""
Llama Backend Module

Watchdog, health probing, and recovery logic specific to llama-server.

Uses a lazy server import (_srv()) and shared utilities from the parent
backend_health module.
"""

import asyncio
import subprocess
import time

import httpx


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    from ..backend_health import _srv as _shared_srv
    return _shared_srv()


# ===================================================================
# GPU-wedge detection (LP-0MS91DHQ9003EGB0)
#
# A wedged llama-server is still HTTP-reachable but its GPU kernels never
# complete: the GPU reads ~100% busy while every slot is idle and no tokens
# are produced. The existing watchdog/health probes only catch dead or
# unreachable workers, so the wedge was invisible to self-healing. Detection
# is proxy-side only (no llama.cpp changes):
#
#   signature = gpu_busy_percent >= threshold AND all slots idle
#
# sustained for ``llama_gpu_wedge_idle_checks_required`` consecutive health
# loop iterations. GPU busy is read from sysfs (AMD) with a rocm-smi
# fallback; when neither is available detection is skipped.
# ===================================================================


def _gpu_busy_percent_sysfs() -> float | None:
    """Read GPU busy % from sysfs (AMD gfx cards). None when unavailable."""
    try:
        with open("/sys/class/drm/card0/device/gpu_busy_percent") as f:
            val = f.read().strip()
        return float(val)
    except Exception:
        return None


def _gpu_busy_percent_rocm_smi() -> float | None:
    """Fallback: parse ``rocm-smi --showuse`` output. None when unavailable."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse"],
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
        for line in out.splitlines():
            if "GPU use (%)" in line:
                val = line.split(":")[-1].strip().rstrip("%")
                return float(val)
    except Exception:
        return None
    return None


def _read_gpu_busy_percent() -> float | None:
    """Best-effort GPU busy %: sysfs first, then rocm-smi."""
    val = _gpu_busy_percent_sysfs()
    if val is None:
        val = _gpu_busy_percent_rocm_smi()
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


def _gpu_wedge_signature(busy_percent: float | None, slots_idle: bool, threshold: float) -> bool:
    """True when the wedge signature is present: high GPU busy + all slots idle."""
    return busy_percent is not None and slots_idle and busy_percent >= threshold


async def _probe_gpu_wedge(
    host: str, port: int, timeout: float = 5.0
) -> tuple[float | None, bool]:
    """Return ``(gpu_busy_percent, all_slots_idle)`` for a model instance."""
    busy = _read_gpu_busy_percent()
    slots_idle = False
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        try:
            resp = await client.get(f"http://{host}:{port}/slots", timeout=timeout)
            if resp.status_code == 200:
                slots_idle = _slots_all_idle(resp.json())
        finally:
            await client.aclose()
    except Exception:
        pass
    return busy, slots_idle


# ===================================================================
# Router self-healing
# ===================================================================

async def _attempt_router_self_heal() -> bool:
    """Attempt router-mode self-healing with capped exponential backoff."""
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
    remaining = max_attempts - len(attempts)

    try:
        for local_attempt in range(remaining):
            attempt_started = time.time()
            attempts.append(attempt_started)
            srv.backend_recovery_state["attempt_timestamps"] = attempts
            attempt_number = len(attempts)

            srv.logger.warning(
                "self-heal attempt %s/%s started (window=%ss)",
                attempt_number,
                max_attempts,
                window_seconds,
            )

            try:
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

                raise RuntimeError("wait_for_llama_server returned False")
            except Exception as exc:
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

            if proc is None:
                if router_mode and not srv.backend_ready:
                    # Probe the backend before attempting restart
                    reachable = await srv._probe_backend_reachable(llama_port)
                    if reachable:
                        srv.logger.info(
                            "watchdog: llama_process is None but backend is reachable on port %d, "
                            "resetting backend_ready to True",
                            llama_port,
                        )
                        srv.backend_ready = True
                        srv.backend_recovery_state["last_failure"] = None
                        continue

                    srv.logger.warning(
                        "watchdog: llama_process is None and backend unreachable, "
                        "attempting restart"
                    )
                    recovered = await srv._attempt_router_self_heal()
                    srv.logger.info(
                        "watchdog restart-from-none recovered=%s", recovered
                    )
                continue

            code = None
            try:
                code = proc.poll()
            except Exception:
                code = None

            worker_unhealthy = False
            if code is None:
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

            if code is None and worker_unhealthy:
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
                srv.logger.error(
                    "watchdog detected llama-server exit code=%s model=%s",
                    code,
                    srv.current_model,
                )

            # LP-0MRCQW0HC000J4F9: Before marking backend_ready=False and
            # triggering full restart, probe whether the backend is still
            # reachable (independent host llama-server in router mode).
            if router_mode:
                reachable = await srv._probe_backend_reachable(llama_port)
                if reachable:
                    srv.logger.info(
                        "watchdog: process exited but backend reachable on port %d, "
                        "resetting backend_ready to True (process exited code=%s)",
                        llama_port,
                        code,
                    )
                    srv.backend_ready = True
                    srv.llama_process = None
                    srv.current_model = None
                    srv.backend_recovery_state["last_failure"] = None
                    continue

            srv.backend_ready = False
            srv._record_backend_signal("other_failures")
            srv.llama_process = None
            srv.current_model = None

            if router_mode:
                recovered = await srv._attempt_router_self_heal()
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

            # GPU-wedge detection configuration (LP-0MS91DHQ9003EGB0)
            gpu_wedge_enabled = bool(
                server_cfg.get("llama_gpu_wedge_detection_enabled", True)
            )
            gpu_wedge_busy_threshold = _coerce_float(
                server_cfg.get("llama_gpu_wedge_busy_threshold_percent", 90.0),
                90.0,
            )
            gpu_wedge_idle_checks = max(
                1,
                _coerce_int(
                    server_cfg.get("llama_gpu_wedge_idle_checks_required", 2),
                    2,
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
                    # may be wedged (busy with idle slots, no progress).
                    # Skip embeddings instances and when no GPU busy source exists.
                    if gpu_wedge_enabled and "--embeddings" not in args:
                        busy_pct, slots_idle = await _probe_gpu_wedge(
                            router_host, port, timeout=probe_timeout
                        )
                        if busy_pct is None:
                            if not gpu_wedge_source_logged:
                                srv.logger.warning(
                                    "model_health: GPU busy source unavailable "
                                    "(no sysfs gpu_busy_percent, no rocm-smi); "
                                    "GPU-wedge detection disabled"
                                )
                                gpu_wedge_source_logged = True
                        elif _gpu_wedge_signature(busy_pct, slots_idle, gpu_wedge_busy_threshold):
                            count = wedge_checks.get(model_id, 0) + 1
                            wedge_checks[model_id] = count
                            if count >= gpu_wedge_idle_checks:
                                srv.logger.error(
                                    "model_health: GPU wedge detected for model %s "
                                    "(port %d): gpu_busy=%.0f%% with all slots idle "
                                    "for %d consecutive check(s); triggering recovery",
                                    model_id,
                                    port,
                                    busy_pct,
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
                                    reason=f"gpu_wedge (busy {busy_pct:.0f}%)",
                                )
                                wedge_checks[model_id] = 0
                                port_first_seen_at[model_id] = time.time()
                            else:
                                srv.logger.warning(
                                    "model_health: possible GPU wedge for model %s "
                                    "(port %d): gpu_busy=%.0f%% with all slots idle "
                                    "(%d/%d checks)",
                                    model_id,
                                    port,
                                    busy_pct,
                                    count,
                                    gpu_wedge_idle_checks,
                                )
                        else:
                            wedge_checks[model_id] = 0
                            if getattr(srv, "gpu_wedge_detected", False):
                                srv.logger.info(
                                    "model_health: GPU wedge cleared for model %s "
                                    "(gpu_busy=%.0f%%)",
                                    model_id,
                                    busy_pct if busy_pct is not None else -1.0,
                                )
                                srv.gpu_wedge_detected = False
                                srv.gpu_wedge_detected_at = None
                                srv.gpu_wedge_last_model = None
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
