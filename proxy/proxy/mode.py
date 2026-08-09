"""Operating-mode state (fast/cheap) for the LLM proxy server.

The proxy runs in one of two operator-selected operating modes:

- **fast** — cloud-backed: remote providers are eligible and the server
  behaves as before (current day settings; ``config-fast.yaml``).
- **cheap** — local-only: requests use only the local llama-server at no
  cost (1-slot pool; ``config-cheap.yaml``).

The active mode is persisted in a small runtime state file
(``proxy/.mode``); when absent the mode defaults to ``fast`` (current
behavior). ``scripts/start-proxy.sh`` reads the mode at startup and selects
the corresponding config file; ``load_config()`` (``proxy/proxy/utils.py``)
falls back to the mode-selected config when ``LLAMA_PROXY_CONFIG`` is unset.

Switching modes via ``POST /admin/set-mode`` persists the new mode and
triggers a full proxy restart (``scripts/start-proxy.sh --restart``) so the
new config profile takes effect. A mode-switch restart terminates in-flight
requests — clients retry (same semantics as slot-schedule transitions,
LP-0MSF9RUSQ007M346). This is accepted behavior, not a bug.
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger("llama-proxy")

MODE_FAST = "fast"
MODE_CHEAP = "cheap"
VALID_MODES = (MODE_FAST, MODE_CHEAP)

DEFAULT_MODE = MODE_FAST

# Mode -> config file name (relative to the proxy directory).
MODE_CONFIG_FILES = {
    MODE_FAST: "config-fast.yaml",
    MODE_CHEAP: "config-cheap.yaml",
}

# Delay (seconds) before the background restart spawns, so the API response
# flushes before the process is killed (LP-0MSLMYEEU002IBH6).
RESTART_DELAY_SECONDS = 1.5

# Serializes set-mode calls and guards the pending-restart flag so a second
# switch cannot arm a second restart while one is already in flight
# (avoids restart loops).
_mode_lock = threading.Lock()
_restart_pending = False


def proxy_dir() -> Path:
    """Return the proxy directory (parent of the ``proxy`` package)."""
    return Path(__file__).parent.parent


def mode_state_file() -> Path:
    """Path to the persisted mode state file (``proxy/.mode``)."""
    return proxy_dir() / ".mode"


def read_mode() -> str:
    """Return the persisted operating mode, defaulting to ``fast``.

    A missing, empty, or invalid state file yields ``fast`` (the current
    behavior when no mode has ever been persisted).
    """
    try:
        text = mode_state_file().read_text(encoding="utf-8").strip().lower()
    except FileNotFoundError:
        return DEFAULT_MODE
    except OSError:
        logger.warning("Failed to read mode state file, defaulting to %s", DEFAULT_MODE)
        return DEFAULT_MODE
    return text if text in VALID_MODES else DEFAULT_MODE


def write_mode(mode: str) -> None:
    """Persist the operating mode to the state file.

    Raises ``ValueError`` for anything other than ``fast`` or ``cheap``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode!r}")
    mode_state_file().write_text(mode.strip().lower() + "\n", encoding="utf-8")


def mode_config_file(mode: str) -> Path:
    """Return the config file path for *mode*.

    ``fast`` → ``config-fast.yaml``, ``cheap`` → ``config-cheap.yaml``.
    Falls back to ``proxy/config.yaml`` when the mode-specific file is
    missing (or the mode is invalid), keeping config.yaml as the
    default/fallback profile.
    """
    name = MODE_CONFIG_FILES.get(mode)
    path = proxy_dir() / name if name else proxy_dir() / "config.yaml"
    return path if path.is_file() else proxy_dir() / "config.yaml"


def resolve_config_path() -> Path:
    """Resolve the active config path.

    Precedence:
    1. ``LLAMA_PROXY_CONFIG`` env var (explicit override — set by
       ``scripts/start-proxy.sh`` from the persisted mode).
    2. The mode-selected config file (``config-fast.yaml`` /
       ``config-cheap.yaml``) when a valid mode has been persisted.
    3. ``proxy/config.yaml`` (default/fallback, current behavior).
    """
    env = os.environ.get("LLAMA_PROXY_CONFIG")
    if env:
        return Path(env)
    return mode_config_file(read_mode())


def restart_pending() -> bool:
    """Whether a mode-switch restart is pending (thread-safe)."""
    with _mode_lock:
        return _restart_pending


def set_mode(mode: str) -> tuple[str, bool]:
    """Persist *mode* and arm a background restart when it changes.

    Returns ``(persisted_mode, restart_triggered)``:

    - Requesting the mode that is already active is a **noop**: nothing is
      persisted and no restart is armed.
    - Requesting a different mode persists the new mode and spawns the
      restart (``scripts/start-proxy.sh --restart``) in the background.

    Raises ``RuntimeError`` when a mode-switch restart is already pending
    and the requested mode differs (rejected to avoid restart loops).
    """
    global _restart_pending
    with _mode_lock:
        if _restart_pending:
            if read_mode() == mode:
                return mode, False
            raise RuntimeError("A mode-switch restart is already in progress")
        if read_mode() == mode:
            return mode, False
        write_mode(mode)
        _restart_pending = True
    _spawn_restart()
    return mode, True


def _spawn_restart() -> None:
    """Spawn ``scripts/start-proxy.sh --restart`` in the background.

    Runs in a daemon thread after ``RESTART_DELAY_SECONDS`` so the API
    response flushes before the process is killed. The persisted mode is
    already written, so a failed restart still applies on the next manual
    start.
    """

    def _run() -> None:
        try:
            time.sleep(RESTART_DELAY_SECONDS)
            script = proxy_dir() / "scripts" / "start-proxy.sh"
            subprocess.Popen(
                ["bash", str(script), "--restart"],
                cwd=str(proxy_dir()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info("Mode-switch restart spawned: %s --restart", script)
        except Exception:
            logger.exception("Failed to spawn mode-switch restart")

    threading.Thread(target=_run, daemon=True).start()
