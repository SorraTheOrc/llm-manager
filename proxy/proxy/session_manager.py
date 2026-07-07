"""
Session management for incremental prompt ingestion.

Provides a SessionManager that tracks per-session message history and
supports hybrid session ID generation (client-supplied X-Session-Id
header or proxy-generated UUID v4).

Sessions expire after a configurable TTL of inactivity (default 3 hours)
and are evicted from memory by a background cleanup task.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json

logger = logging.getLogger("llama-proxy")

# Default session inactivity TTL in seconds (3 hours)
DEFAULT_SESSION_TTL_SECONDS = 3 * 60 * 60


@dataclass
class Session:
    """Represents a single conversation session."""

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    # Stores the list of messages (role, content) tuples for delta computation
    messages: List[Dict[str, Any]] = field(default_factory=list)
    # Tracks whether the session has been invalidated (e.g. due to message edit)
    invalidated: bool = False
    # Set to True after explicit backend restore evidence is observed.
    restore_confirmed: bool = False

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity_at

    def touch(self) -> None:
        """Update last_activity_at to now."""
        self.last_activity_at = time.monotonic()

    def is_expired(self, ttl: float) -> bool:
        """Return True if this session has been idle longer than ttl seconds."""
        return self.idle_seconds > ttl


class SessionManager:
    """In-memory session registry with TTL-based eviction.

    Usage::

        manager = SessionManager(ttl_seconds=10800)
        # Create or retrieve a session
        session = manager.get_or_create("client-provided-id")
        # Update session with new messages
        manager.update_messages(session.session_id, messages)
    """

    def __init__(self, ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS, cleanup_interval_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval_seconds: float = cleanup_interval_seconds
        # Metrics
        self._sessions_created: int = 0
        self._sessions_expired: int = 0

    # ----------------------------------------------------------------
    # Session lifecycle
    # ----------------------------------------------------------------

    async def get_or_create(
        self, session_id: Optional[str] = None
    ) -> Tuple[Session, bool]:
        """Return an existing session or create a new one.

        Args:
            session_id: Client-supplied session ID. If None, a UUID v4
                is generated.

        Returns:
            A tuple ``(session, created)`` where *created* is True when
            a new session was created.
        """
        async with self._lock:
            if session_id is not None:
                existing = self._sessions.get(session_id)
                if existing is not None:
                    if existing.is_expired(self.ttl_seconds):
                        # Session expired – evict and create fresh
                        del self._sessions[session_id]
                        self._sessions_expired += 1
                        logger.info(
                            f"Session {session_id} expired (idle "
                            f"{existing.idle_seconds:.0f}s), evicting"
                        )
                    elif existing.invalidated:
                        # Session was invalidated (e.g. message edit) –
                        # remove and create fresh
                        del self._sessions[session_id]
                        self._sessions_expired += 1
                        logger.info(
                            f"Session {session_id} invalidated, creating new"
                        )
                    else:
                        # Valid session – touch and return
                        existing.touch()
                        return existing, False

            # Create new session
            new_id = session_id if session_id is not None else str(uuid.uuid4())
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            self._sessions_created += 1
            logger.info(f"Created new session {new_id[:8]}...")
            return session, True

    async def get(self, session_id: str) -> Optional[Session]:
        """Return a session by ID without creating one.

        Returns None if the session does not exist or has expired.
        Expired sessions are evicted.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self.ttl_seconds) or session.invalidated:
                del self._sessions[session_id]
                self._sessions_expired += 1
                return None
            session.touch()
            return session

    async def invalidate(self, session_id: str) -> bool:
        """Mark a session as invalidated so the next request creates a fresh one.

        Returns True if the session was found and invalidated, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.invalidated = True
            logger.info(f"Session {session_id[:8]}... invalidated")
            return True

    async def remove(self, session_id: str) -> bool:
        """Remove a session from the registry.

        Returns True if the session existed and was removed.
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all active session IDs with their metadata.

        Returns a list of dicts with session info including
        ``session_id``, ``created_at``, ``last_activity_at``,
        ``response_time`` (ISO8601), ``message_count``,
        ``idle_seconds``, ``age_seconds``, ``invalidated``, and
        ``restore_confirmed``.

        Expired sessions are evicted before listing.
        """
        now = time.monotonic()
        wall_now = time.time()
        monotonic_offset = wall_now - now
        sessions: List[Dict[str, Any]] = []
        async with self._lock:
            expired_ids = []
            for sid, session in self._sessions.items():
                if now - session.last_activity_at > self.ttl_seconds:
                    expired_ids.append(sid)
                else:
                    # Convert monotonic timestamps to wall-clock ISO8601 for sorting
                    wall_last = monotonic_offset + session.last_activity_at
                    wall_created = monotonic_offset + session.created_at
                    sessions.append({
                        "session_id": sid,
                        "created_at": session.created_at,
                        "last_activity_at": session.last_activity_at,
                        "response_time": datetime.fromtimestamp(wall_last, tz=timezone.utc).isoformat(timespec="seconds"),
                        "message_count": session.message_count,
                        "idle_seconds": round(session.idle_seconds, 1),
                        "age_seconds": round(session.age_seconds, 1),
                        "invalidated": session.invalidated,
                        "restore_confirmed": session.restore_confirmed,
                    })
            for sid in expired_ids:
                del self._sessions[sid]
                self._sessions_expired += 1
        return sessions

    # ----------------------------------------------------------------
    # Message history management
    # ----------------------------------------------------------------

    def compute_delta(
        self,
        existing_messages: List[Dict[str, Any]],
        incoming_messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Compute the delta (new messages) between existing and incoming.

        The incoming message list must start with the same messages as the
        existing history. Any messages that extend beyond the existing
        history are the delta.

        If the incoming messages do not match the existing history prefix
        (indicating an edit), returns the entire incoming messages and
        marks the session for invalidation.

        Returns:
            A tuple ``(delta_messages, history_matches)`` where
            *delta_messages* is the list of new messages and
            *history_matches* is True if the prefix matched.
        """
        if not existing_messages:
            # No prior history – entire incoming is the delta
            return list(incoming_messages), True

        if not incoming_messages:
            return [], True

        # Check that the incoming messages start with the existing history
        if len(incoming_messages) < len(existing_messages):
            # Incoming is shorter than existing – can't be a prefix match.
            # This means the conversation was edited.
            return list(incoming_messages), False

        # Compare prefix
        for i, existing_msg in enumerate(existing_messages):
            incoming_msg = incoming_messages[i]
            if (
                existing_msg.get("role") != incoming_msg.get("role")
                or existing_msg.get("content") != incoming_msg.get("content")
            ):
                # Mismatch – history was edited
                return list(incoming_messages), False

        # Prefix matches – delta is the remaining messages
        delta = incoming_messages[len(existing_messages) :]
        return delta, True

    def compute_delta_metrics(
        self,
        existing_messages: List[Dict[str, Any]],
        incoming_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return deterministic payload metrics for full vs delta ingestion.

        The output is intended for tests/diagnostics and can be used to
        quantify payload reduction independently from backend restore behavior.
        """
        delta_messages, history_matches = self.compute_delta(
            existing_messages, incoming_messages
        )

        # Stable JSON serialization for byte-size comparisons.
        full_payload_bytes = len(
            json.dumps(incoming_messages, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        delta_payload_bytes = len(
            json.dumps(delta_messages, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )

        reduction_ratio = 0.0
        if full_payload_bytes > 0:
            reduction_ratio = max(0.0, min(1.0, 1.0 - (delta_payload_bytes / full_payload_bytes)))

        reason: Optional[str] = None
        mode = "delta"
        if not history_matches:
            mode = "full"
            reason = "history_mismatch"
            reduction_ratio = 0.0
        elif not existing_messages:
            mode = "full"
            reason = "no_existing_history"
            reduction_ratio = 0.0
        elif not delta_messages:
            mode = "delta"
            reason = "no_new_messages"

        return {
            "history_matches": history_matches,
            "mode": mode,
            "fallback_reason": reason,
            "full_payload_bytes": full_payload_bytes,
            "delta_payload_bytes": delta_payload_bytes,
            "reduction_ratio": reduction_ratio,
            "reduction_percent": round(reduction_ratio * 100.0, 2),
            "delta_message_count": len(delta_messages),
            "incoming_message_count": len(incoming_messages),
        }

    async def update_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Update the session's message history with the full message list.

        This replaces the stored history with the provided messages and
        increments the message count.

        Returns True if the session was found and updated, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.messages = list(messages)
            session.message_count = len(messages)
            session.touch()
            return True

    async def set_restore_confirmed(self, session_id: str, confirmed: bool) -> bool:
        """Update strict restore-confirmed state for a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.restore_confirmed = bool(confirmed)
            session.touch()
            return True

    async def append_messages(
        self,
        session_id: str,
        new_messages: List[Dict[str, Any]],
    ) -> bool:
        """Append new messages to the session's history.

        Returns True if the session was found and updated, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.messages.extend(new_messages)
            session.message_count = len(session.messages)
            session.touch()
            return True

    # ----------------------------------------------------------------
    # Cleanup and metrics
    # ----------------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Evict all expired and invalidated sessions.

        Returns the number of sessions evicted.
        """
        async with self._lock:
            expired_ids = [
                sid
                for sid, session in self._sessions.items()
                if session.is_expired(self.ttl_seconds) or session.invalidated
            ]
            for sid in expired_ids:
                del self._sessions[sid]
            self._sessions_expired += len(expired_ids)
            if expired_ids:
                logger.info(
                    f"Evicted {len(expired_ids)} expired/invalidated session(s)"
                )
            return len(expired_ids)

    async def _cleanup_loop(self) -> None:
        """Background task that periodically evicts expired sessions."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval_seconds)
                evicted = await self.cleanup_expired()
                if evicted:
                    logger.info(
                        f"Background cleanup evicted {evicted} session(s)"
                    )
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception:
            logger.exception("Unexpected error in session cleanup loop")

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task.

        Safe to call multiple times; only starts one task.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                logger.info("Session cleanup task started")
            except RuntimeError:
                logger.warning(
                    "No running event loop; session cleanup task not started"
                )

    def stop_cleanup_task(self) -> None:
        """Cancel the background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Session cleanup task stopped")

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions (excluding expired)."""
        return len(self._sessions)

    @property
    def total_sessions_created(self) -> int:
        return self._sessions_created

    @property
    def total_sessions_expired(self) -> int:
        return self._sessions_expired

    def get_metrics(self) -> Dict[str, Any]:
        """Return session metrics as a dict for observability."""
        return {
            "sessions_active": self.active_session_count,
            "sessions_created_total": self.total_sessions_created,
            "sessions_expired_total": self.total_sessions_expired,
        }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session info as a dict without creating it.

        Returns None if session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_activity_at": session.last_activity_at,
            "message_count": session.message_count,
            "idle_seconds": round(session.idle_seconds, 1),
            "age_seconds": round(session.age_seconds, 1),
            "invalidated": session.invalidated,
            "restore_confirmed": session.restore_confirmed,
        }