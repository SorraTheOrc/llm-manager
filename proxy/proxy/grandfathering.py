"""Session→model grandfathering registry for fast↔cheap mode switches.

When the proxy switches operating mode (fast ↔ cheap), models whose remote
access is cut by the new mode (e.g. ``github``, or the opencode/opencode-go/
deepseek tiers of the hybrid ``plan``/``author``/``code`` models) become
unavailable. Sessions that were **already active before the switch** should
keep using their model — this module persists which model each session was
using so that routing can continue to serve grandfathered sessions after the
mode-switch restart (which wipes the in-memory SessionManager).

Semantics
---------

- A **binding** is ``(session_id → model, last_seen, recorded_mode)``.
- A binding is **grandfathered** iff it is still *valid* (not expired), its
  model is *restricted* by the current mode (see ``model_is_restricted``),
  and it was *recorded in a different mode* than the currently active one
  (i.e. the session had prior activity before the current mode period).
- **Expiry** (earliest of):
  - the next scheduled mode transition (from ``ModeScheduleConfig``), or
  - the session going idle (``session_ttl``, default 3 h), or
  - the fallback grace window (``grace_seconds``, default = session TTL)
    when the mode schedule is disabled.

Known limitations / OPEN QUESTION
---------------------------------

- A session that starts in cheap mode, uses a remote model during the
  following fast period, and then hits another cheap window is *not*
  grandfathered (its binding's ``recorded_mode`` matches cheap again). This
  is conservative (never leaks remote calls) at the cost of that marginal
  session. Refining eligibility to track per-mode usage is deferred to the
  routing feature (LP-0MSMFQ8SQ008B52E).
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from proxy.mode import ModeScheduleConfig
from proxy.session_manager import DEFAULT_SESSION_TTL_SECONDS

logger = logging.getLogger("llama-proxy")

# Fallback grace window when the mode schedule is disabled (default: session TTL).
DEFAULT_GRACE_SECONDS = DEFAULT_SESSION_TTL_SECONDS

# Registry state file version; bump on incompatible format changes.
_REGISTRY_VERSION = 1


@dataclass
class Binding:
    """A single session→model binding.

    Attributes:
        model: The resolved model name the session was using.
        last_seen: Epoch-seconds timestamp of the session's last request.
        recorded_mode: The operating mode (``fast``/``cheap``) active when the
            binding was created. A binding is eligible for grandfathering only
            when ``recorded_mode`` differs from the currently active mode.
    """

    model: str
    last_seen: float
    recorded_mode: str


def proxy_dir() -> Path:
    """Return the proxy directory (parent of the ``proxy`` package)."""
    return Path(__file__).parent.parent


def default_state_file() -> Path:
    """Path to the persisted registry state file (beside ``proxy/.mode``)."""
    return proxy_dir() / "grandfathering-state.json"


def next_mode_transition(
    schedule: ModeScheduleConfig, now: datetime
) -> datetime:
    """Return the datetime of the next scheduled mode transition after *now*.

    Entries are ``(HH:MM, mode)`` sorted by time (never empty for an enabled
    schedule — ``ModeScheduleConfig`` falls back to the built-in schedule).
    The next transition is the first entry strictly after ``now``; when none
    remains today it wraps to the first entry tomorrow.
    """
    now_time = now.time()
    entries = schedule.entries
    for entry in entries:
        if entry.time > now_time:
            return now.replace(
                hour=entry.time.hour,
                minute=entry.time.minute,
                second=0,
                microsecond=0,
            )
    first = entries[0]
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(
        hour=first.time.hour,
        minute=first.time.minute,
        second=0,
        microsecond=0,
    )


def _remote_provider_count(models: dict[str, Any], model: str) -> int:
    """Count the remote providers of *model* in *models* (0 when absent)."""
    model_cfg = models.get(model)
    if not isinstance(model_cfg, dict):
        return 0
    providers = model_cfg.get("providers") or []
    return sum(
        1
        for p in providers
        if isinstance(p, dict) and p.get("type") == "remote"
    )


def model_is_restricted(
    model: str,
    current_models: dict[str, Any],
    other_models: dict[str, Any],
) -> bool:
    """Return True when *model*'s remote access was cut by a mode switch.

    A model is **restricted** by the current mode when either:

    - it is absent from the current-mode config (e.g. ``github`` in cheap
      mode), or
    - it is present but has **fewer remote providers** than the other-mode
      config (e.g. ``plan``/``author``/``code`` in cheap mode keep only their
      local provider, losing the opencode/opencode-go/deepseek tiers).

    Models with identical (or more) remote coverage in the current mode are
    not restricted.
    """
    if model not in current_models:
        return True
    return _remote_provider_count(current_models, model) < _remote_provider_count(
        other_models, model
    )


class GrandfatheringRegistry:
    """Persisted session→model binding registry with expiry bookkeeping.

    The registry is deliberately independent of the in-memory
    ``SessionManager``: it is keyed by session id and survives proxy
    restarts, so bindings recorded before a mode-switch restart are
    available afterwards.

    Args:
        state_file: Path of the JSON state file (loaded at construction when
            present; missing/corrupt files start an empty registry).
        session_ttl: Idle timeout after which a binding expires (seconds).
        grace_seconds: Fallback grace window when the mode schedule is
            disabled (seconds; default = session_ttl).
        mode_schedule: The active ``ModeScheduleConfig``, or ``None`` when
            the schedule is disabled. Expiry then uses ``grace_seconds``.
        enabled: Master switch. When False the registry still records, but
            ``is_grandfathered`` always returns False.
    """

    def __init__(
        self,
        state_file: str | Path | None = None,
        *,
        session_ttl: float = DEFAULT_SESSION_TTL_SECONDS,
        grace_seconds: float | None = None,
        mode_schedule: ModeScheduleConfig | None = None,
        enabled: bool = True,
    ) -> None:
        self.state_file = Path(state_file) if state_file else default_state_file()
        self.session_ttl = float(session_ttl)
        self.grace_seconds = (
            float(grace_seconds)
            if grace_seconds is not None
            else float(DEFAULT_GRACE_SECONDS)
        )
        self.mode_schedule = mode_schedule
        self.enabled = enabled
        self._bindings: dict[str, Binding] = {}
        self._load()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str | None,
        model: str | None,
        mode: str,
        now: float | None = None,
    ) -> bool:
        """Record/refresh the binding for *session_id*.

        Anonymous sessions (missing/empty session id or model) are never
        recorded. An existing binding keeps its ``recorded_mode`` (the mode
        in which the session was first seen), so a session that predates the
        current mode remains eligible for grandfathering as it reconnects.

        Returns True when a binding was created/refreshed.
        """
        if not session_id or not model:
            return False
        now = float(now) if now is not None else time.time()
        existing = self._bindings.get(session_id)
        recorded_mode = existing.recorded_mode if existing else mode
        self._bindings[session_id] = Binding(
            model=model, last_seen=now, recorded_mode=recorded_mode
        )
        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Binding | None:
        """Return the binding for *session_id*, or None."""
        return self._bindings.get(session_id)

    def __len__(self) -> int:
        """Number of recorded bindings."""
        return len(self._bindings)

    def deadline(self, session_id: str, now: float | None = None) -> float | None:
        """Return the epoch-second deadline of the binding, or None.

        The deadline is the earliest of the next scheduled mode transition
        and the session-idle timeout (``last_seen + session_ttl``). When the
        mode schedule is disabled, the fallback grace window
        (``last_seen + grace_seconds``) is used instead of the transition.
        """
        binding = self._bindings.get(session_id)
        if binding is None:
            return None
        idle_deadline = binding.last_seen + self.session_ttl
        if self._schedule_enabled():
            # The transition bound is computed from the binding's recorded
            # time so the deadline is STABLE (independent of the evaluation
            # time): a binding first seen at 01:00 always expires at the
            # first transition after 01:00, never later.
            transition = next_mode_transition(
                self.mode_schedule, datetime.fromtimestamp(binding.last_seen)
            )
            return min(idle_deadline, transition.timestamp())
        return min(idle_deadline, binding.last_seen + self.grace_seconds)

    def is_valid(self, session_id: str, now: float | None = None) -> bool:
        """True when the binding exists and has not expired at *now*."""
        deadline = self.deadline(session_id, now)
        if deadline is None:
            return False
        now = float(now) if now is not None else time.time()
        return now < deadline

    def is_grandfathered(
        self,
        session_id: str,
        current_mode: str,
        current_models: dict[str, Any],
        other_models: dict[str, Any],
        now: float | None = None,
    ) -> bool:
        """True when *session_id* may keep using its model under *current_mode*.

        Grandfathered iff: the feature is enabled, a valid (unexpired)
        binding exists, the binding was recorded in a different mode than the
        currently active one (prior activity before this mode period), and
        the bound model is restricted by the current mode.
        """
        if not self.enabled:
            return False
        binding = self._bindings.get(session_id)
        if binding is None:
            return False
        if binding.recorded_mode == current_mode:
            return False
        if not self.is_valid(session_id, now):
            return False
        return model_is_restricted(binding.model, current_models, other_models)

    # ------------------------------------------------------------------
    # Pruning / persistence
    # ------------------------------------------------------------------

    def prune(self, now: float | None = None) -> int:
        """Drop expired bindings and return the number removed."""
        now = float(now) if now is not None else time.time()
        expired = [
            sid
            for sid in self._bindings
            if not self.is_valid(sid, now)
        ]
        for sid in expired:
            del self._bindings[sid]
        return len(expired)

    def save(self, path: str | Path | None = None) -> None:
        """Persist the registry atomically (tmp file + rename).

        A missing state file is created; an empty registry persists cleanly.
        """
        target = Path(path) if path else self.state_file
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "bindings": {
                sid: {
                    "model": b.model,
                    "last_seen": b.last_seen,
                    "recorded_mode": b.recorded_mode,
                }
                for sid, b in sorted(self._bindings.items())
            },
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_enabled(self) -> bool:
        return self.mode_schedule is not None and self.mode_schedule.enabled

    def _load(self) -> None:
        """Load bindings from the state file; missing/corrupt → empty."""
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning(
                "grandfathering: ignoring unreadable state file %s",
                self.state_file,
            )
            return
        if not isinstance(raw, dict):
            logger.warning(
                "grandfathering: ignoring malformed state file %s",
                self.state_file,
            )
            return
        bindings = raw.get("bindings")
        if not isinstance(bindings, dict):
            return
        for sid, entry in bindings.items():
            if not isinstance(entry, dict):
                continue
            model = entry.get("model")
            last_seen = entry.get("last_seen")
            recorded_mode = entry.get("recorded_mode")
            if (
                isinstance(model, str)
                and isinstance(last_seen, (int, float))
                and isinstance(recorded_mode, str)
            ):
                self._bindings[sid] = Binding(
                    model=model,
                    last_seen=float(last_seen),
                    recorded_mode=recorded_mode,
                )


