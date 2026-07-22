"""
TTS Backend Module

Watchdog, health probing, and recovery logic specific to the TTS server.

Uses a lazy server import (_srv()) and shared utilities from the parent
backend_health module.
"""

import asyncio
import time


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    from ..backend_health import _srv as _shared_srv
    return _shared_srv()


# ===================================================================
# TTS configuration helpers
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


# ===================================================================
# TTS recovery snapshot
# ===================================================================


def _tts_recovery_snapshot() -> dict:
    """Return a snapshot of the TTS recovery state for the /health endpoint."""
    srv = _srv()
    state = dict(srv.tts_recovery_state)
    attempts = state.get("attempt_timestamps")
    state["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
    return state


# ===================================================================
# TTS self-healing
# ===================================================================


async def _attempt_tts_self_heal() -> bool:
    """Attempt TTS server self-healing with capped retry attempts.

    Follows a similar but simpler pattern to llama-server self-healing:
    configurable max attempts within a time window, with clear logging.
    """
    from ..backend_health import (
        _get_tts_self_heal_max_attempts,
        _get_tts_self_heal_window,
        _prune_recovery_attempts,
    )

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


# ===================================================================
# TTS watchdog loop
# ===================================================================


async def _tts_watchdog_loop() -> None:
    """Periodically monitor the TTS server and restart if dead.

    Respects ``tts_enabled`` config flag — when disabled, the loop
    exits immediately and does no monitoring.

    When the TTS process has exited, triggers ``_attempt_tts_self_heal()``
    which manages retry backoff and max-attempts tracking.
    """
    from ..backend_health import _attempt_tts_self_heal, _get_tts_watchdog_interval

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
