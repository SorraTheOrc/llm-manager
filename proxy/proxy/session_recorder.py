"""
Session Recorder Module

Captures raw client-proxy-provider message payloads to disk for later
debugging, auditing, and analysis. Provides always-on, non-blocking
recording of all message traffic flowing through the proxy.

Recording is organized on disk by session ID:

    <recording-path>/
        <session-id>/
            <timestamp>-request.json
            <timestamp>-proxy_to_provider-request.json
            <timestamp>-response.json

Each JSON file wraps the payload with metadata (session_id, direction,
timestamp) so files can be inspected individually without external context.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("llama-proxy.session_recorder")

# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------

DIR_CLIENT_TO_PROXY = "client_to_proxy"
DIR_PROXY_TO_PROVIDER = "proxy_to_provider"
DIR_PROVIDER_TO_CLIENT = "provider_to_client"

VALID_DIRECTIONS = {DIR_CLIENT_TO_PROXY, DIR_PROXY_TO_PROVIDER, DIR_PROVIDER_TO_CLIENT}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RECORDING_PATH = "proxy/session-recordings/"

# ---------------------------------------------------------------------------
# SessionRecorder
# ---------------------------------------------------------------------------


class SessionRecorder:
    """Records session message payloads to disk in a non-blocking manner.

    Attributes:
        recording_path: Absolute or relative path to the root recording
            directory. Defaults to ``proxy/session-recordings/``.
    """

    def __init__(self, recording_path: str = DEFAULT_RECORDING_PATH):
        """Initialize the recorder and ensure the recording directory exists.

        Args:
            recording_path: Filesystem path for storing recordings.
                Created automatically if it does not exist.
        """
        # Strip trailing slash for consistent path matching
        self.recording_path = recording_path.rstrip("/")

        # Ensure the root directory exists
        try:
            Path(self.recording_path).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to create recording directory %s: %s",
                self.recording_path, e,
            )

    # ------------------------------------------------------------------
    # Factory from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, default_path: str = DEFAULT_RECORDING_PATH) -> "SessionRecorder":
        """Create a SessionRecorder from a proxy config dict.

        Looks for a ``session_recording`` section with an optional ``path``
        key. Falls back to *default_path* if no path is configured.

        Args:
            config: The proxy configuration dictionary (read from config.yaml).
            default_path: Default recording path when config lacks the key.

        Returns:
            A new SessionRecorder instance.
        """
        sr_cfg = config.get("session_recording", {}) if isinstance(config, dict) else {}
        path = sr_cfg.get("path", default_path) if isinstance(sr_cfg, dict) else default_path
        return cls(recording_path=path)

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    async def record_request(
        self,
        session_id: str,
        direction: str,
        payload: Any,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Record a request payload to disk (non-blocking).

        Args:
            session_id: Unique session identifier.
            direction: One of ``"client_to_proxy"``, ``"proxy_to_provider"``,
                or ``"provider_to_client"``.
            payload: The request payload to record (must be JSON-serializable).
            model: Optional model name to include in recording metadata.
            provider: Optional provider name to include in recording metadata.

        Returns:
            The absolute file path of the written recording, or ``None`` if
            the write failed.
        """
        return await self._write_recording(session_id, direction, payload, suffix="request", model=model, provider=provider)

    async def record_response(
        self,
        session_id: str,
        direction: str,
        payload: Any,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Record a response payload to disk (non-blocking).

        Accepts only fully-assembled response payloads (not individual
        SSE chunks). Assembly must be done by the caller before calling
        this method.

        Args:
            session_id: Unique session identifier.
            direction: One of ``"client_to_proxy"``, ``"proxy_to_provider"``,
                or ``"provider_to_client"``.
            payload: The assembled response payload to record.
            model: Optional model name to include in recording metadata.
            provider: Optional provider name to include in recording metadata.

        Returns:
            The absolute file path of the written recording, or ``None`` if
            the write failed.
        """
        return await self._write_recording(session_id, direction, payload, suffix="response", model=model, provider=provider)

    async def _write_recording(
        self,
        session_id: str,
        direction: str,
        payload: Any,
        suffix: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Core recording method — serialize, build path, write to disk.

        Uses ``asyncio.to_thread`` to perform the synchronous file write
        on a thread pool executor, keeping the event loop responsive.

        Args:
            session_id: Unique session identifier.
            direction: Recording direction constant.
            payload: The payload to record (must be JSON-serializable).
            suffix: ``"request"`` or ``"response"``.
            model: Optional model name to include in recording metadata.
            provider: Optional provider name to include in recording metadata.
        """
        if direction not in VALID_DIRECTIONS:
            logger.warning("Invalid recording direction: %s", direction)
            return None

        # Build the session directory path with sanitisation
        session_id_safe = self._sanitise_session_id(session_id)
        session_dir = Path(self.recording_path) / session_id_safe

        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to create session directory %s: %s",
                session_dir, e,
            )
            return None

        # Build filename with timestamp
        timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        filename = f"{timestamp}-{suffix}.json"
        filepath = session_dir / filename

        # Prepare the record envelope (include model/provider when available)
        record = {
            "session_id": session_id,
            "direction": direction,
            "timestamp": timestamp,
            "payload": payload,
        }
        if model:
            record["model"] = model
        if provider:
            record["provider"] = provider

        # Serialize to JSON
        try:
            json_bytes = json.dumps(record, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to serialize recording for session %s: %s",
                session_id, e,
            )
            return None

        # Write to disk in a thread (non-blocking)
        try:
            await asyncio.to_thread(self._write_file, filepath, json_bytes)
        except OSError as e:
            logger.warning(
                "Failed to write recording for session %s: %s",
                session_id, e,
            )
            return None

        return str(filepath)

    @staticmethod
    def _write_file(path: Path, data: bytes) -> None:
        """Synchronous file write — runs in a thread pool executor."""
        path.write_bytes(data)

    # ------------------------------------------------------------------
    # Query / retrieval methods
    # ------------------------------------------------------------------

    def get_recordings_list(self, session_id: str) -> List[Dict[str, Any]]:
        """Return metadata for all recording files of a session.

        Returns a list of dicts, each containing:
            - filename: Base filename of the recording.
            - timestamp: ISO8601 timestamp extracted from the file content.
            - direction: Recording direction extracted from the file content.
            - file_size: Size of the file in bytes.

        Returns an empty list if the session directory does not exist or
        contains no recording files.
        """
        session_dir = Path(self.recording_path) / self._sanitise_session_id(session_id)
        try:
            if not session_dir.is_dir():
                return []
        except OSError:
            # Permission error or inaccessible directory
            return []

        recordings: List[Dict[str, Any]] = []
        try:
            for entry in sorted(session_dir.iterdir()):
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".json"):
                    continue
                # Read just enough to extract metadata
                try:
                    content = json.loads(entry.read_bytes())
                    recordings.append({
                        "filename": entry.name,
                        "timestamp": content.get("timestamp", ""),
                        "direction": content.get("direction", ""),
                        "file_size": entry.stat().st_size,
                    })
                except (json.JSONDecodeError, OSError):
                    # Skip corrupted/unreadable files
                    continue
        except OSError as e:
            logger.warning(
                "Failed to list recordings for session %s: %s",
                session_id, e,
            )
            return []

        return recordings

    def get_recording(self, session_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full content of a single recording file.

        Path traversal protection: rejects filenames containing path
        separators or parent-directory references.

        Args:
            session_id: The session identifier.
            filename: The base filename of the recording (e.g.,
                ``"2026-07-06T10:00:00.000000-request.json"``).

        Returns:
            The parsed JSON content of the recording, or ``None`` if the
            file does not exist, is corrupted, or the filename is invalid.
        """
        # Path traversal protection
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            return None

        session_dir = Path(self.recording_path) / self._sanitise_session_id(session_id)
        filepath = session_dir / filename

        try:
            if not filepath.exists() or not filepath.is_file():
                return None
            return json.loads(filepath.read_bytes())
        except (json.JSONDecodeError, OSError):
            return None

    def list_sessions_by_model(self, model: str) -> List[Dict[str, Any]]:
        """Return session IDs that have recordings for a specific model.

        Scans session directories and checks recording metadata for the
        given model name. Returns a list of dicts with session_id and
        timestamp of the most recent recording.

        Args:
            model: The model name to filter by.

        Returns:
            A list of dicts with ``session_id`` and ``last_activity`` keys,
            sorted by most recent activity first.
        """
        if not model:
            return []

        base = Path(self.recording_path)
        if not base.is_dir():
            return []

        sessions: List[Dict[str, Any]] = []
        try:
            for entry in sorted(base.iterdir()):
                if not entry.is_dir():
                    continue
                sid = entry.name
                last_ts = ""
                found = False
                for f in entry.iterdir():
                    if not f.is_file() or not f.name.endswith(".json"):
                        continue
                    try:
                        content = json.loads(f.read_bytes())
                        if content.get("model") == model:
                            found = True
                            ts = content.get("timestamp", "")
                            if ts > last_ts:
                                last_ts = ts
                    except (json.JSONDecodeError, OSError):
                        continue
                if found:
                    sessions.append({
                        "session_id": sid,
                        "last_activity": last_ts,
                    })
        except OSError as e:
            logger.warning("Failed to list sessions by model %s: %s", model, e)
            return []

        # Sort by last_activity descending
        sessions.sort(key=lambda s: s["last_activity"], reverse=True)
        return sessions

    def list_sessions(self) -> List[str]:
        """Return all session IDs that have recording directories.

        Sessions are sorted alphabetically. Only directories that contain
        at least one ``.json`` file are included.
        """
        base = Path(self.recording_path)
        if not base.is_dir():
            return []

        sessions: List[str] = []
        try:
            for entry in sorted(base.iterdir()):
                if not entry.is_dir():
                    continue
                # Check if the directory has at least one .json file
                has_json = any(
                    f.is_file() and f.name.endswith(".json")
                    for f in entry.iterdir()
                )
                if has_json:
                    sessions.append(entry.name)
        except OSError as e:
            logger.warning(
                "Failed to list sessions in %s: %s",
                self.recording_path, e,
            )
            return []

        return sessions

    @staticmethod
    def _sanitise_session_id(session_id: str) -> str:
        """Sanitise a session ID for use as a directory name.

        Replaces path separator characters (``/``, ``\\``) with underscores
        to prevent directory traversal.
        """
        return session_id.replace("/", "_").replace("\\", "_")
