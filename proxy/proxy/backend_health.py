"""
Backend Health Module

Backend recovery, self-healing, watchdog monitoring, and worker-health
functions extracted from the monolithic lifecycle.py.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import subprocess
import time

import httpx
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backend-recovery helpers
# ===================================================================

def _self_heal_retry_after_seconds() -> int:
    srv = _srv()
    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    configured = server_cfg.get(
        "llama_self_heal_retry_after_seconds",
        srv.backend_recovery_state.get("retry_after_seconds", 30),
    )
    try:
        retry_after = int(configured or 30)
    except Exception:
        retry_after = 30
    retry_after = max(1, retry_after)
    srv.backend_recovery_state["retry_after_seconds"] = retry_after
    return retry_after


def _is_self_healing_active() -> bool:
    srv = _srv()
    try:
        return bool(srv.backend_recovery_state.get("in_progress"))
    except Exception:
        return False


def _self_healing_response(path: str) -> JSONResponse:
    srv = _srv()
    retry_after = srv._self_heal_retry_after_seconds()
    message = (
        "Backend error detected, team is working on recovery. "
        "Please retry after 30 seconds."
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "backend_recovery",
                "code": "backend_recovery_in_progress",
                "message": message,
            },
            "status": 503,
            "retry_after": retry_after,
            "path": f"/{path.lstrip('/')}",
        },
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _backend_recovery_snapshot() -> dict:
    srv = _srv()
    state = dict(srv.backend_recovery_state)
    attempts = state.get("attempt_timestamps")
    state["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
    return state


# ===================================================================
# Worker-health helper
# ===================================================================

def _worker_process_unhealthy(proc: subprocess.Popen | None) -> bool:
    """Detect unhealthy llama worker states (for example zombie children)."""
    srv = _srv()
    if proc is None or srv.psutil is None:
        return False

    pid = getattr(proc, "pid", None)
    if not pid:
        return False

    try:
        parent = srv.psutil.Process(pid)
        children = parent.children(recursive=True)
    except Exception:
        return False

    zombie_statuses = {"zombie", "dead"}
    try:
        zombie_statuses.add(str(srv.psutil.STATUS_ZOMBIE).lower())
    except Exception:
        pass
    try:
        zombie_statuses.add(str(srv.psutil.STATUS_DEAD).lower())
    except Exception:
        pass

    for child in children:
        try:
            status = str(child.status()).lower()
        except Exception:
            continue
        if status in zombie_statuses:
            srv.logger.error(
                "watchdog detected unhealthy worker pid=%s status=%s parent_pid=%s",
                getattr(child, "pid", "unknown"),
                status,
                pid,
            )
            return True

    return False


def _prune_recovery_attempts(
    attempts: list[float], now_ts: float, window_seconds: int
) -> list[float]:
    window = max(1, int(window_seconds))
    return [float(ts) for ts in attempts if now_ts - float(ts) <= window]


def _coerce_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


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

def _extract_model_port_from_args(args: list) -> int | None:
    """Extract the model instance port from its argument list.

    The args list comes from the router's /models endpoint status.args
    field and contains things like ``--port 41649``.
    """
    if not isinstance(args, list):
        return None
    try:
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                return int(args[i + 1])
    except (ValueError, IndexError):
        pass
    return None


async def _router_model_health_loop() -> None:
    """Periodically check loaded models' reachability in router mode.

    Guardrails to reduce false positives:
    - legacy interval-key fallback (llama_health_check_interval)
    - initial grace window after model (re)load or port change
    - multi-attempt probing before counting a failure
    - consecutive failure threshold before unload/reload
    """
    srv = _srv()

    # Stateful counters across loop iterations
    consecutive_failures: dict[str, int] = {}
    observed_ports: dict[str, int] = {}
    port_first_seen_at: dict[str, float] = {}

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

                srv.logger.info("model_health: unloading dead model %s", model_id)
                try:
                    client = (
                        srv._http_client
                        if srv._http_client
                        else httpx.AsyncClient(timeout=10.0)
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


# ===================================================================
# TTS self-healing watchdog
# ===================================================================


def _get_tts_watchdog_interval(server_cfg: dict) -> float:
    """Return the TTS watchdog check interval in seconds."""
    try:
        return float(server_cfg.get("tts_self_heal_interval_seconds", 10.0) or 10.0)
    except Exception:
        return 10.0


def _get_tts_self_heal_max_attempts(server_cfg: dict) -> int:
    """Return the max TTS self-heal restart attempts."""
    try:
        return int(server_cfg.get("tts_self_heal_max_attempts", 3) or 3)
    except Exception:
        return 3


def _get_tts_self_heal_window(server_cfg: dict) -> int:
    """Return the TTS self-heal window in seconds."""
    try:
        return int(server_cfg.get("tts_self_heal_window_seconds", 120) or 120)
    except Exception:
        return 120


def _get_tts_self_heal_probe_timeout(server_cfg: dict) -> float:
    """Return the TTS health probe timeout in seconds."""
    try:
        return float(server_cfg.get("tts_self_heal_probe_timeout_seconds", 3.0) or 3.0)
    except Exception:
        return 3.0


def _tts_recovery_snapshot() -> dict:
    """Return a snapshot of the TTS recovery state for the /health endpoint."""
    srv = _srv()
    state = dict(srv.tts_recovery_state)
    attempts = state.get("attempt_timestamps")
    state["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
    return state


async def _attempt_tts_self_heal() -> bool:
    """Attempt TTS server self-healing with capped retry attempts.

    Follows a similar but simpler pattern to llama-server self-healing:
    configurable max attempts within a time window, with clear logging.
    """
    srv = _srv()

    server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}
    max_attempts = _get_tts_self_heal_max_attempts(server_cfg)
    window_seconds = _get_tts_self_heal_window(server_cfg)

    now_ts = time.time()
    attempts = list(srv.tts_recovery_state.get("attempt_timestamps", []))
    attempts = _prune_recovery_attempts(attempts, now_ts, window_seconds)

    srv.tts_recovery_state["attempt_timestamps"] = attempts
    srv.tts_recovery_state["max_attempts"] = max_attempts
    srv.tts_recovery_state["window_seconds"] = window_seconds

    if len(attempts) >= max_attempts:
        srv.tts_recovery_state["in_progress"] = False
        srv.tts_recovery_state["last_failure"] = (
            f"TTS self-heal throttled: max {max_attempts} attempts in "
            f"{window_seconds}s"
        )
        srv.logger.error(
            "TTS self-heal giving up: max attempts reached (%s attempts in %ss); "
            "manual intervention required",
            max_attempts,
            window_seconds,
        )
        return False

    srv.tts_recovery_state["in_progress"] = True
    remaining = max_attempts - len(attempts)

    try:
        for local_attempt in range(remaining):
            attempt_started = time.time()
            attempts.append(attempt_started)
            srv.tts_recovery_state["attempt_timestamps"] = attempts
            attempt_number = len(attempts)

            srv.logger.warning(
                "TTS self-heal attempt %s/%s started (window=%ss)",
                attempt_number,
                max_attempts,
                window_seconds,
            )

            try:
                from proxy.lifecycle import start_tts_server

                restarted = start_tts_server()
                if restarted is None:
                    raise RuntimeError("start_tts_server returned None")

                srv.tts_process = restarted

                # Verify TTS becomes healthy after restart
                tts_port = int(server_cfg.get("tts_server_port", 8081) or 8081)
                from proxy.lifecycle import wait_for_tts_server

                ready = await wait_for_tts_server(tts_port, timeout=30)
                if ready:
                    srv.tts_recovery_state["last_failure"] = None
                    srv.tts_recovery_state["in_progress"] = False
                    srv.logger.info(
                        "TTS self-heal succeeded on attempt %s/%s",
                        attempt_number,
                        max_attempts,
                    )
                    return True

                raise RuntimeError("wait_for_tts_server returned False")
            except Exception as exc:
                srv.tts_recovery_state["last_failure"] = str(exc)
                srv.logger.error(
                    "TTS self-heal attempt %s/%s failed: %s",
                    attempt_number,
                    max_attempts,
                    exc,
                )

            if local_attempt < remaining - 1:
                # Simple backoff: base 2s, double each attempt
                delay = 2.0 * (2**local_attempt)
                srv.logger.warning(
                    "TTS self-heal backoff sleeping %.1fs before retry", delay
                )
                await asyncio.sleep(delay)

        srv.logger.error(
            "TTS self-heal exhausted after %s attempt(s) within %ss; "
            "manual intervention required",
            remaining,
            window_seconds,
        )
        return False
    finally:
        srv.tts_recovery_state["in_progress"] = False


async def _tts_watchdog_loop() -> None:
    """Periodically monitor the TTS server and restart if dead.

    Respects ``tts_enabled`` config flag — when disabled, the loop
    exits immediately and does no monitoring.

    When the TTS process has exited, triggers ``_attempt_tts_self_heal()``
    which manages retry backoff and max-attempts tracking.
    """
    srv = _srv()

    while True:
        try:
            server_cfg = srv.config.get("server", {}) if isinstance(srv.config, dict) else {}

            # Check if TTS is enabled — exit immediately if disabled
            tts_enabled = bool(server_cfg.get("tts_enabled", True))
            if not tts_enabled:
                srv.logger.info(
                    "TTS watchdog: TTS is disabled in config, watchdog not starting"
                )
                return

            interval = _get_tts_watchdog_interval(server_cfg)
            await asyncio.sleep(interval)

            proc = srv.tts_process

            # Check if process has exited
            if proc is None:
                srv.logger.warning(
                    "TTS watchdog: tts_process is None, attempting restart"
                )
                await _attempt_tts_self_heal()
                continue

            code = None
            try:
                code = proc.poll()
            except Exception:
                code = None

            if code is None:
                # Process is still running, nothing to do
                continue

            # Process has exited
            srv.logger.error(
                "TTS watchdog: TTS server exited with code=%s", code
            )
            await _attempt_tts_self_heal()

        except asyncio.CancelledError:
            srv.logger.info("TTS watchdog loop cancelled")
            return
        except Exception:
            srv.logger.exception("TTS watchdog loop error")

