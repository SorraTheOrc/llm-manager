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
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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

# Maximum number of sessions returned by list_sessions() and related
# endpoints. Prevents the web UI dropdown from being overwhelmed with
# hundreds of stale sessions.
MAX_SESSION_DROPDOWN_COUNT = 15

# Maximum number of entries kept in the shared metadata index per recording
# path. Bounds memory usage of the index; when exceeded the least-recently-
# active session is evicted (LP-0MSNM94A1006WAOJ). Configurable via
# ``session_recording.max_index_entries`` in config.yaml.
DEFAULT_MAX_INDEX_ENTRIES = 1000

# Maximum number of session directories visited by a cold-start scan when
# the in-memory metadata index is empty. Bounds worst-case scan cost so a
# cold /admin/sessions call never re-reads the full recordings tree
# (LP-0MSNKMZCP003T8OG).
COLD_SCAN_DIR_LIMIT = 50

# Retention window for session recordings, in days. Recordings older than
# this are pruned by the background prune loop (and at startup). A value
# <= 0 disables pruning entirely (LP-0MT2TC3PB005H1RD).
DEFAULT_RETENTION_DAYS = 3

# How often the background prune loop re-runs, in seconds (LP-0MT2TC3PB005H1RD).
DEFAULT_PRUNE_INTERVAL_SECONDS = 3600

# ---------------------------------------------------------------------------
# Shared metadata index (module-level, keyed by recording path)
# ---------------------------------------------------------------------------
#
# The write path (proxy/router_helpers.py) constructs a fresh SessionRecorder
# for every request batch, while the UI reads through a cached instance
# (proxy/ui.py ``_get_recorder``). A per-instance index would therefore be
# invisible to the UI. The index is shared at module level, keyed by the
# recording path, so writes and reads observe the same state.
#
# Entry shape matches ``_extract_session_preview`` output:
#     session_id, response_time, last_activity, model, provider, preview_text
_SHARED_INDEX: dict[str, dict[str, dict[str, Any]]] = {}
_SHARED_INDEX_LOCK = threading.RLock()

# Recording paths whose shared index has already been warmed (populated via
# write updates or a cold scan). Guards against re-scanning on every call.
_INDEX_WARM_PATHS: set[str] = set()

# Observability counters for the shared index, keyed by recording path
# (LP-0MSNM9IAC000GVXT).
_INDEX_OBS: dict[str, dict[str, Any]] = {}
_OBS_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# SessionRecorder
# ---------------------------------------------------------------------------


class SessionRecorder:
    """Records session message payloads to disk in a non-blocking manner.

    Attributes:
        recording_path: Absolute or relative path to the root recording
            directory. Defaults to ``proxy/session-recordings/``.
    """

    def __init__(
        self,
        recording_path: str = DEFAULT_RECORDING_PATH,
        max_index_entries: int = DEFAULT_MAX_INDEX_ENTRIES,
        cold_scan_dir_limit: int = COLD_SCAN_DIR_LIMIT,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        prune_interval_seconds: int = DEFAULT_PRUNE_INTERVAL_SECONDS,
    ):
        """Initialize the recorder and ensure the recording directory exists.

        Args:
            recording_path: Filesystem path for storing recordings.
                Created automatically if it does not exist.
            max_index_entries: Maximum number of entries in the shared
                metadata index for this recording path. Oldest sessions are
                evicted when the index exceeds this size.
            cold_scan_dir_limit: Maximum number of session directories
                visited by the cold-start scan when the index is empty.
            retention_days: Recordings older than this many days are pruned
                by the background prune loop. Values <= 0 disable pruning.
            prune_interval_seconds: How often the background prune loop
                re-runs.
        """
        # Strip trailing slash for consistent path matching
        self.recording_path = recording_path.rstrip("/")
        self.max_index_entries = max_index_entries
        self.cold_scan_dir_limit = cold_scan_dir_limit
        self.retention_days = retention_days
        self.prune_interval_seconds = prune_interval_seconds

        # Shared metadata index for this recording path (module-level, so
        # writer instances and the UI's cached instance see the same state).
        with _SHARED_INDEX_LOCK:
            self._index = _SHARED_INDEX.setdefault(
                self.recording_path, {}
            )

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
        max_entries = (
            sr_cfg.get("max_index_entries", DEFAULT_MAX_INDEX_ENTRIES)
            if isinstance(sr_cfg, dict)
            else DEFAULT_MAX_INDEX_ENTRIES
        )
        cold_scan_limit = (
            sr_cfg.get("cold_scan_dir_limit", COLD_SCAN_DIR_LIMIT)
            if isinstance(sr_cfg, dict)
            else COLD_SCAN_DIR_LIMIT
        )
        retention_days = (
            sr_cfg.get("retention_days", DEFAULT_RETENTION_DAYS)
            if isinstance(sr_cfg, dict)
            else DEFAULT_RETENTION_DAYS
        )
        prune_interval = (
            sr_cfg.get("prune_interval_seconds", DEFAULT_PRUNE_INTERVAL_SECONDS)
            if isinstance(sr_cfg, dict)
            else DEFAULT_PRUNE_INTERVAL_SECONDS
        )
        return cls(
            recording_path=path,
            max_index_entries=max_entries,
            cold_scan_dir_limit=cold_scan_limit,
            retention_days=retention_days,
            prune_interval_seconds=prune_interval,
        )

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    async def record_request(
        self,
        session_id: str,
        direction: str,
        payload: Any,
        model: str | None = None,
        provider: str | None = None,
    ) -> str | None:
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
        model: str | None = None,
        provider: str | None = None,
    ) -> str | None:
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
        model: str | None = None,
        provider: str | None = None,
    ) -> str | None:
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
        timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
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

        # Update the shared metadata index so list_sessions* can serve the
        # session without re-reading the recordings tree (LP-0MSNM90VD0030GEL).
        self._update_index(
            session_id, direction, timestamp, payload, model, provider,
        )

        return str(filepath)

    @staticmethod
    def _write_file(path: Path, data: bytes) -> None:
        """Synchronous file write — runs in a thread pool executor."""
        path.write_bytes(data)

    # ------------------------------------------------------------------
    # Retention pruning
    # ------------------------------------------------------------------

    async def prune_older_than(self, days: int) -> dict[str, Any]:
        """Delete recordings older than *days* days (non-blocking).

        Runs the synchronous scan in a thread pool executor so the event
        loop stays responsive while potentially large trees are walked.
        When *days* <= 0 pruning is disabled and nothing is deleted.

        Safety: the scan only descends into immediate subdirectories of the
        configured recording root. Symlinked files/directories are never
        followed, so content reachable through a symlink outside the root is
        never deleted. Only files matching the recorder naming convention
        (``*-request.json`` / ``*-response.json``) are considered; other
        files in session directories are left untouched.

        Age determination (oldest-first preference, LP-0MT2TC3PB005H1RD):
        1. ISO timestamp embedded in the filename;
        2. ``timestamp`` field in the recorded JSON envelope;
        3. file mtime as a last resort.

        Args:
            days: Retention window in days. Recordings older than this are
                deleted. Values <= 0 disable pruning (no-op).

        Returns:
            A stats dict with keys: ``disabled``, ``files_scanned``,
            ``files_deleted``, ``dirs_removed``, and ``errors``.
        """
        return await asyncio.to_thread(self._prune_older_than_sync, days)

    def _prune_older_than_sync(self, days: int) -> dict[str, Any]:
        """Synchronous pruning scan — runs in a thread pool executor."""
        stats = {
            "disabled": False,
            "files_scanned": 0,
            "files_deleted": 0,
            "dirs_removed": 0,
            "errors": 0,
            "cutoff": None,
        }
        if days <= 0:
            stats["disabled"] = True
            return stats

        cutoff = datetime.now(UTC) - timedelta(days=days)
        stats["cutoff"] = cutoff.isoformat(timespec="seconds")

        root = Path(self.recording_path)
        if not root.is_dir():
            return stats

        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            # Never follow symlinked session directories out of the root.
            if child.is_symlink():
                continue
            try:
                self._prune_session_dir(child, cutoff, stats)
            except OSError:
                stats["errors"] += 1
        return stats

    def _prune_session_dir(self, session_dir: Path, cutoff: datetime, stats: dict) -> None:
        """Delete recordings in *session_dir* older than *cutoff*.

        Removes the directory itself when it becomes empty. Only recording
        files matching the recorder naming convention are touched.
        """
        for f in sorted(session_dir.iterdir()):
            if not f.is_file():
                continue
            if f.is_symlink():
                # Never touch content reachable through symlinks.
                continue
            if not (f.name.endswith("-request.json") or f.name.endswith("-response.json")):
                continue
            stats["files_scanned"] += 1
            ts = self._recording_file_timestamp(f)
            if ts is not None and ts < cutoff:
                try:
                    f.unlink()
                    stats["files_deleted"] += 1
                except OSError:
                    stats["errors"] += 1

        # Remove the session directory when it holds no recording files.
        try:
            remaining = [
                e for e in session_dir.iterdir()
                if e.is_file()
                and (e.name.endswith("-request.json") or e.name.endswith("-response.json"))
            ]
            if not remaining:
                session_dir.rmdir()
                stats["dirs_removed"] += 1
        except OSError:
            # Directory not empty (scratch files present) or permission issue.
            pass

    def _recording_file_timestamp(self, filepath: Path) -> datetime | None:
        """Best-effort timestamp for a recording file.

        Prefers the ISO timestamp embedded in the filename, falls back to
        the ``timestamp`` envelope field, then to file mtime. Returns None
        only when no timestamp can be derived at all.
        """
        m = self._FILENAME_TS_RE.match(filepath.name)
        if m:
            try:
                return self._coerce_utc(datetime.fromisoformat(m.group(1)))
            except ValueError:
                pass
        try:
            content = json.loads(filepath.read_bytes())
            ts = content.get("timestamp") if isinstance(content, dict) else None
            if isinstance(ts, str):
                try:
                    return self._coerce_utc(datetime.fromisoformat(ts))
                except ValueError:
                    pass
        except (json.JSONDecodeError, OSError):
            pass
        try:
            return datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC)
        except OSError:
            return None

    @staticmethod
    def _coerce_utc(dt: datetime) -> datetime:
        """Normalise a parsed timestamp to an aware UTC datetime.

        Older recordings embedded naive ISO timestamps (no tz offset); the
        mtime fallback is always aware. Comparisons against the UTC cutoff
        require both sides to be aware, so naive values are assumed UTC
        (LP-0MT2TC3PB005H1RD).
        """
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    # ------------------------------------------------------------------
    # Metadata index (shared, per recording path)
    # ------------------------------------------------------------------

    def _update_index(
        self,
        session_id: str,
        direction: str,
        timestamp: str,
        payload: Any,
        model: str | None,
        provider: str | None,
    ) -> None:
        """Update the shared metadata index after a successful write.

        Entry fields mirror ``_extract_session_preview``: session_id,
        response_time (first provider_to_client response), last_activity
        (latest recording), model/provider (first non-None seen), and
        preview_text (first client_to_proxy user message).
        """
        with _SHARED_INDEX_LOCK:
            entry = self._index.get(session_id)
            if entry is None:
                entry = {
                    "session_id": session_id,
                    "response_time": "",
                    "last_activity": "",
                    "model": None,
                    "provider": None,
                    "preview_text": "",
                }
                self._index[session_id] = entry

            # last_activity = latest recording timestamp
            if not entry["last_activity"] or timestamp > entry["last_activity"]:
                entry["last_activity"] = timestamp

            # response_time = first provider_to_client response
            if direction == DIR_PROVIDER_TO_CLIENT and not entry["response_time"]:
                entry["response_time"] = timestamp

            # model/provider: first non-None value wins
            if model and not entry["model"]:
                entry["model"] = model
            if provider and not entry["provider"]:
                entry["provider"] = provider

            # preview_text: first client_to_proxy user message
            if direction == DIR_CLIENT_TO_PROXY and not entry["preview_text"]:
                raw = self._extract_message_text(payload)
                entry["preview_text"] = (
                    self._truncate_preview(raw) if raw else ""
                )

            self._evict_oldest_if_over_cap()

    def _evict_oldest_if_over_cap(self) -> None:
        """Drop the least-recently-active sessions when the index exceeds cap.

        Caller must hold ``_SHARED_INDEX_LOCK``. Keeps the index bounded so
        memory usage does not grow with the full recordings tree.
        """
        while len(self._index) > self.max_index_entries:
            oldest_sid = min(
                self._index,
                key=lambda sid: (
                    self._index[sid].get("last_activity")
                    or self._index[sid].get("response_time")
                    or ""
                ),
            )
            del self._index[oldest_sid]

    def get_index_entry(self, session_id: str) -> dict[str, Any] | None:
        """Return a copy of the shared index entry for *session_id*.

        Returns None if the session has no index entry yet.
        """
        with _SHARED_INDEX_LOCK:
            entry = self._index.get(session_id)
            return dict(entry) if entry is not None else None

    def get_all_index_entries(self) -> list[dict[str, Any]]:
        """Return copies of all shared index entries for this recording path.

        Useful for tests and observability to inspect the full (unbounded by
        the dropdown cap) index contents.
        """
        with _SHARED_INDEX_LOCK:
            return [dict(e) for e in self._index.values()]

    def _ensure_index_warm(self) -> None:
        """Populate the shared index via a bounded cold scan if not yet warm.

        Runs at most once per recording path (per process). Uses only the
        ``cold_scan_dir_limit`` most-recent session directories so a cold
        start never re-reads the full recordings tree.
        """
        with _SHARED_INDEX_LOCK:
            if self.recording_path in _INDEX_WARM_PATHS:
                self._record_index_hit()
                return
            # An index already populated by write-path updates is warm — do
            # not re-scan the recordings tree (LP-0MSNM9BDV007DQXB AC1).
            if self._index:
                _INDEX_WARM_PATHS.add(self.recording_path)
                self._record_index_hit()
                return
            start = time.monotonic()
            self._rebuild_index_from_scan()
            duration = time.monotonic() - start
            _INDEX_WARM_PATHS.add(self.recording_path)
            self._record_cold_scan(duration)

    def _record_index_hit(self) -> None:
        """Count a list_sessions* call served from the warm index."""
        with _OBS_LOCK:
            obs = _INDEX_OBS.setdefault(self.recording_path, {
                "index_size": 0,
                "index_hits": 0,
                "cold_scans": 0,
                "last_scan_duration_seconds": 0.0,
            })
            obs["index_hits"] += 1

    def _record_cold_scan(self, duration: float) -> None:
        """Count a cold-scan fallback and record its duration."""
        with _OBS_LOCK:
            obs = _INDEX_OBS.setdefault(self.recording_path, {
                "index_size": 0,
                "index_hits": 0,
                "cold_scans": 0,
                "last_scan_duration_seconds": 0.0,
            })
            obs["cold_scans"] += 1
            obs["last_scan_duration_seconds"] = duration

    def get_index_observability(self) -> dict[str, Any]:
        """Return index observability counters for this recording path.

        Fields: index_size, index_hits, cold_scans,
        last_scan_duration_seconds. Read-only snapshot; no behaviour change.
        """
        with _OBS_LOCK:
            obs = _INDEX_OBS.get(self.recording_path, {
                "index_size": 0,
                "index_hits": 0,
                "cold_scans": 0,
                "last_scan_duration_seconds": 0.0,
            })
            result = dict(obs)
        with _SHARED_INDEX_LOCK:
            result["index_size"] = len(self._index)
        return result

    def _rebuild_index_from_scan(self) -> None:
        """Cold-start scan: visit the newest cold_scan_dir_limit dirs only.

        Uses the bounded per-dir extractor so each visited dir costs at most
        a couple of small file reads regardless of how many recordings the
        dir holds (LP-0MSNM97PA000XA0M).
        """
        base = Path(self.recording_path)
        if not base.is_dir():
            return
        try:
            dirs = [d for d in base.iterdir() if d.is_dir()]
        except OSError:
            return
        # Newest first by directory mtime
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs[: self.cold_scan_dir_limit]:
            preview = self._extract_session_preview_bounded(d)
            if preview is not None:
                self._index[preview["session_id"]] = preview
        self._evict_oldest_if_over_cap()

    # Filename pattern for recorder files: ``<iso-timestamp>-request.json``
    # / ``<iso-timestamp>-response.json``. The leading ISO timestamp lets us
    # derive last_activity without reading file contents.
    _FILENAME_TS_RE = re.compile(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2})?)"
    )

    def _extract_session_preview_bounded(self, session_dir: Path) -> dict[str, Any] | None:
        """Extract session preview reading at most two recording files.

        Bounded variant of ``_extract_session_preview`` for the cold-start
        scan: a session dir may hold hundreds of files (large sessions are
        multi-GB), so reading every file is exactly the disk-thrash the index
        is meant to avoid. Instead:

        - ``last_activity`` is derived from the newest **filename** timestamp
          (no content read);
        - ``response_time``/model/provider come from the earliest response
          file (read once);
        - ``preview_text`` comes from the earliest request file (read once).

        Returns the same preview dict shape as ``_extract_session_preview``,
        or None when the directory has no valid recording files.
        """
        sid = session_dir.name
        try:
            files = sorted(
                f for f in session_dir.iterdir()
                if f.is_file() and f.name.endswith(".json")
            )
        except OSError:
            return None
        if not files:
            return None

        # Derive last_activity from the newest filename timestamp.
        last_activity = ""
        for f in files:
            m = self._FILENAME_TS_RE.match(f.name)
            if m and m.group(1) > last_activity:
                last_activity = m.group(1)

        # Earliest request file → preview_text (client_to_proxy payload).
        req_file = next(
            (f for f in files if f.name.endswith("-request.json")), None
        )
        preview_text = ""
        first_req_payload: Any = None
        if req_file is not None:
            try:
                req_content = json.loads(req_file.read_bytes())
            except (json.JSONDecodeError, OSError):
                req_content = None
            if isinstance(req_content, dict):
                first_req_payload = req_content.get("payload")
                if not last_activity:
                    last_activity = req_content.get("timestamp", "")

        # Earliest response file → response_time, model, provider.
        resp_file = next(
            (f for f in files if f.name.endswith("-response.json")), None
        )
        response_time = ""
        model: Any = None
        provider: Any = None
        if resp_file is not None:
            try:
                resp_content = json.loads(resp_file.read_bytes())
            except (json.JSONDecodeError, OSError):
                resp_content = None
            if isinstance(resp_content, dict):
                response_time = resp_content.get("timestamp", "")
                model = resp_content.get("model")
                provider = resp_content.get("provider")

        # Fall back to request metadata when no response file exists.
        if resp_file is None and isinstance(req_content, dict):
            response_time = req_content.get("timestamp", "")
            model = req_content.get("model")
            provider = req_content.get("provider")

        if not last_activity:
            last_activity = response_time
        if not last_activity:
            return None

        if first_req_payload is not None:
            raw_text = self._extract_message_text(first_req_payload)
            preview_text = self._truncate_preview(raw_text) if raw_text else ""

        return {
            "session_id": sid,
            "response_time": response_time or "",
            "last_activity": last_activity,
            "model": model,
            "provider": provider,
            "preview_text": preview_text,
        }

    # ------------------------------------------------------------------
    # Query / retrieval methods
    # ------------------------------------------------------------------

    def get_recordings_list(self, session_id: str) -> list[dict[str, Any]]:
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

        recordings: list[dict[str, Any]] = []
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
                        "model": content.get("model", ""),
                        "provider": content.get("provider", ""),
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

    def get_recording(self, session_id: str, filename: str) -> dict[str, Any] | None:
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

    @staticmethod
    @staticmethod
    def _extract_message_text(payload: Any) -> str:
        """Extract the text content from the first user message in a payload.

        Looks for ``payload.messages`` and returns the ``content`` of the
        first message with ``role == "user"``.  If no user message is found
        returns an empty string.

        Args:
            payload: The recording payload, typically a dict with optional
                ``messages`` key.

        Returns:
            The text content of the first user message, or empty string.
        """
        if not isinstance(payload, dict):
            return ""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Content may be an array of content parts
                    texts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    return "\n".join(texts)
                return str(content)
        return ""

    @staticmethod
    def _truncate_preview(text: str, max_chars: int = 80) -> str:
        """Truncate *text* to *max_chars* characters, appending ``...``
        when truncation occurs.
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _extract_session_preview(session_dir: Path) -> dict[str, Any] | None:
        """Extract preview data for a session from its recording files.

        Finds the first ``provider_to_client`` response recording and
        returns its timestamp (as ``response_time``), model, and provider.
        If no response recording exists, falls back to the first
        ``client_to_proxy`` request recording.

        Also determines the **latest** recording timestamp across all
        files in the session and returns it as ``last_activity`` — this
        is the sorting key used to order sessions by most recent update.

        Also extracts a ``preview_text`` — the first 80 characters of
        the user's first message from the earliest ``client_to_proxy``
        recording — so the session list can show a meaningful label.

        Args:
            session_dir: Path to the session's recording directory.

        Returns:
            A dict with ``session_id``, ``response_time``,
            ``last_activity``, ``model``, ``provider``, and
            ``preview_text``, or ``None`` if the directory has no valid
            recording files.
        """
        sid = session_dir.name
        recordings: list[dict[str, Any]] = []
        first_client_payload: Any = None
        try:
            for f in sorted(session_dir.iterdir()):
                if not f.is_file() or not f.name.endswith(".json"):
                    continue
                try:
                    content = json.loads(f.read_bytes())
                    direction = content.get("direction", "")
                    recordings.append({
                        "direction": direction,
                        "timestamp": content.get("timestamp", ""),
                        "model": content.get("model", ""),
                        "provider": content.get("provider", ""),
                    })
                    # Capture the first client_to_proxy payload for preview
                    if direction == "client_to_proxy" and first_client_payload is None:
                        first_client_payload = content.get("payload")
                except (json.JSONDecodeError, OSError):
                    continue
        except OSError:
            return None

        if not recordings:
            return None

        # Find first provider_to_client response
        first_resp = None
        first_req = None
        # Determine the latest recording timestamp across all files
        last_timestamp: str | None = None
        for r in recordings:
            ts = r["timestamp"]
            if last_timestamp is None or ts > last_timestamp:
                last_timestamp = ts
            if r["direction"] == "provider_to_client":
                if first_resp is None or ts < first_resp["timestamp"]:
                    first_resp = r
            if r["direction"] == "client_to_proxy":
                if first_req is None or ts < first_req["timestamp"]:
                    first_req = r

        # Prefer response data, fall back to request data
        source = first_resp or first_req
        if source is None:
            source = recordings[0]

        # Extract preview text from the first client_to_proxy payload
        raw_text = SessionRecorder._extract_message_text(first_client_payload)
        preview_text = SessionRecorder._truncate_preview(raw_text) if raw_text else ""

        return {
            "session_id": sid,
            "response_time": source.get("timestamp", ""),
            "last_activity": last_timestamp or source.get("timestamp", ""),
            "model": source.get("model", ""),
            "provider": source.get("provider", ""),
            "preview_text": preview_text,
        }

    def list_sessions_by_model(self, model: str) -> list[dict[str, Any]]:
        """Return session IDs that have recordings for a specific model.

        Serves from the shared metadata index (warmed lazily via a bounded
        cold scan), so repeated calls do not re-read the recordings tree
        (LP-0MSNKMZCP003T8OG).

        When no recordings with matching model metadata are found (e.g.
        recordings from before the model enrichment field was added), falls
        back to listing all available sessions as unattributed.

        Args:
            model: The model name to filter by.

        Returns:
            A list of dicts with ``session_id``, ``response_time``,
            ``model``, and ``provider``, sorted by most recent activity.
        """
        if not model:
            return []

        self._ensure_index_warm()

        with _SHARED_INDEX_LOCK:
            entries = list(self._index.values())

        model_sessions = [e for e in entries if e.get("model") == model]
        if model_sessions:
            model_sessions.sort(
                key=lambda s: s.get("last_activity") or s.get("response_time") or "",
                reverse=True,
            )
            return model_sessions

        # Fall back to all sessions when no model metadata exists yet
        all_sessions = [e for e in entries if e.get("response_time")]
        all_sessions.sort(
            key=lambda s: s.get("last_activity") or s.get("response_time") or "",
            reverse=True,
        )
        return all_sessions

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all session IDs that have recording directories.

        Serves from the shared metadata index, warmed lazily via a bounded
        cold scan when the index is empty (proxy restart / externally
        written recordings). Repeated calls perform zero recording-file
        reads (LP-0MSNKMZCP003T8OG).

        Returns a list of dicts with ``session_id``, ``response_time``,
        ``model``, and ``provider``, sorted by most recent activity and
        capped at ``MAX_SESSION_DROPDOWN_COUNT``.
        """
        self._ensure_index_warm()

        with _SHARED_INDEX_LOCK:
            sessions = list(self._index.values())

        sessions.sort(
            key=lambda s: s.get("last_activity") or s.get("response_time") or "",
            reverse=True,
        )
        return sessions[:MAX_SESSION_DROPDOWN_COUNT]

    @staticmethod
    def _sanitise_session_id(session_id: str) -> str:
        """Sanitise a session ID for use as a directory name.

        Replaces path separator characters (``/``, ``\\``) with underscores
        to prevent directory traversal.
        """
        return session_id.replace("/", "_").replace("\\", "_")
