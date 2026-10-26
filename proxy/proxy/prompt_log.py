"""Prompt log — append-only structured record of submitted proxy prompts.

Why this exists
---------------
``proxy.log`` is an unstructured, noisy stream: it records that *a* request
happened and dumps a redacted body preview, but there is no compact, queryable
record of *what was submitted* and *where it was routed*. Session recordings
capture full payloads, but they live in per-session directories that are hard
to scan. This module writes one JSONL line per handled chat/completions
request (time, first-message prompt, session ID, requested model and the
model that actually served it after fallback routing), which the ``/logs``
Web UI surfaces in its **Prompt Log** tab (:mod:`proxy.ui`,
``GET /logs/prompt-log``).

Contract
--------
* **One entry per handled HTTP request.** The hook lives in
  ``ui.proxy_openai_api`` *outside* the request coalescer
  (``proxy.request_coalescer``), so a coalesced duplicate HTTP request also
  produces its own entry (AC1 is stated per HTTP request); both entries report
  the same routed model because the duplicate is served the leader's response
  (and therefore its ``X-Resolved-Model`` header).
* **Exactly one complete line, written once.** The entry is appended after
  dispatch resolved, so the requested *and* routed model are both known: there
  is no in-place line mutation and never a partially written line.
* **Non-blocking and never fatal.** :func:`record_exchange` only *enqueues* a
  serialised line; a daemon writer thread performs the file I/O. Append
  failures are counted (``write_errors``) and logged rate-limited — they never
  propagate to the request path.
* **Bounded.** Prompt text is capped by ``prompt_log.max_prompt_chars`` (with a
  ``truncated`` flag); the UI window reads at most ``prompt_log.max_entries``
  lines from a bounded tail of the file (never the whole file); the file itself
  rotates to ``prompt-log.jsonl.1`` once it exceeds
  ``prompt_log.max_file_bytes``.
* **System prompts stay out.** The recorded prompt is the first **non-system**
  message (chat) or the ``prompt`` field (completions) — the same redaction
  policy as the existing body preview in
  :func:`proxy.router_helpers._get_request_preview`, whose extraction this
  module reuses (:func:`proxy.router_helpers._first_message_content`).
* **Routed model = the existing resolution convention.** The value is taken
  from the response's ``X-Resolved-Model`` header
  (``local/<llama_model>`` for local dispatch, ``<provider>/<model>`` for a
  remote/fallback dispatch), falling back to ``X-Provider``. Requests rejected
  before routing record ``error:<code>`` (e.g. ``error:startup_ramp``,
  ``error:model_loading``, ``error:unknown_model``).

Session IDs use the shared :func:`proxy.session._resolve_session_id_header`
chain (``x-session-id`` → ``session_id`` → ``x-client-request-id`` →
``x-session-affinity``), falling back to ``"unknown"``, so the log matches the
session IDs shown elsewhere in the UI.

Out of scope (deliberately): embeddings/TTS requests, ``/v1/models``, search
features and backfilling history from existing logs.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: File name used inside the configured log directory.
FILENAME = "prompt-log.jsonl"

#: Suffix appended to the rotated files (``prompt-log.jsonl.1``, ``.2``, …).
ROTATED_SUFFIX = ".1"

DEFAULT_MAX_PROMPT_CHARS = 2000
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB
DEFAULT_MAX_ROTATED_FILES = 3

#: Paths (as passed to ``proxy_openai_api``) that are recorded. Embeddings and
#: ``/v1/models`` are explicitly out of scope.
RECORDED_PATHS: frozenset[str] = frozenset({"chat/completions", "completions"})

#: Bounded queue between the request coroutine and the writer thread. A full
#: queue drops entries (counted) rather than blocking the request path.
QUEUE_MAXSIZE = 1000

#: Backwards read chunk size and hard cap for the bounded tail read.
TAIL_CHUNK_BYTES = 64 * 1024
MAX_TAIL_BYTES = 8 * 1024 * 1024

#: Minimum spacing between rate-limited append-error log lines.
ERROR_LOG_INTERVAL_SECONDS = 60.0

logger = logging.getLogger("llama-proxy")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptLogConfig:
    """Resolved ``prompt_log:`` configuration."""

    enabled: bool = True
    path: Path = Path(FILENAME)
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_rotated_files: int = DEFAULT_MAX_ROTATED_FILES


#: Bounded rotation: at most this many ``.N`` backup files are kept (shifting
#: older ones down and dropping the oldest), so total on-disk growth stays
#: ``<= (1 + max_rotated_files) * max_file_bytes``.
MAX_ROTATED_FILES = DEFAULT_MAX_ROTATED_FILES


def resolve_config(config: Mapping[str, Any] | None, log_dir: Any = None) -> PromptLogConfig:
    """Resolve the ``prompt_log:`` section into a :class:`PromptLogConfig`.

    Args:
        config: The effective proxy config mapping (may be ``None``).
        log_dir: The resolved log directory (``server.log_dir``). The log file
            defaults to ``<log_dir>/prompt-log.jsonl``; the package ``logs/``
            directory is used when no log directory is available. An explicit
            ``prompt_log.path`` always wins.

    Returns:
        The resolved configuration. Invalid *values* are clamped/normalised by
        the validators; this resolver assumes validation already passed.
    """
    section = (config or {}).get("prompt_log") or {}
    if not isinstance(section, Mapping):
        section = {}

    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"

    raw_path = section.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(os.path.expanduser(raw_path.strip()))
    else:
        path = Path(log_dir) / FILENAME

    return PromptLogConfig(
        enabled=bool(section.get("enabled", True)),
        path=path,
        max_prompt_chars=int(section.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS)),
        max_entries=int(section.get("max_entries", DEFAULT_MAX_ENTRIES)),
        max_file_bytes=int(section.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)),
        max_rotated_files=int(
            section.get("max_rotated_files", DEFAULT_MAX_ROTATED_FILES)
        ),
    )


def validate_prompt_log_config(config: Mapping[str, Any] | None) -> list[str]:
    """Validate the ``prompt_log:`` config section.

    Returns a list of human-readable problems (empty when valid). Mirrors the
    ``validate_compaction_config`` convention: the caller
    (``proxy.utils._validate_prompt_log_config``) turns problems into a
    startup ``ValueError``.
    """
    problems: list[str] = []
    config = config or {}
    section = config.get("prompt_log")
    if section is None:
        return problems
    if not isinstance(section, Mapping):
        return ["prompt_log: must be a mapping"]

    if "enabled" in section and not isinstance(section["enabled"], bool):
        problems.append("prompt_log.enabled: must be a boolean")

    path = section.get("path")
    if path is not None and not isinstance(path, str):
        problems.append("prompt_log.path: must be a string")

    for key, minimum in (
        ("max_prompt_chars", 1),
        ("max_entries", 1),
        ("max_rotated_files", 1),
    ):
        if key not in section:
            continue
        value = section[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            problems.append(f"prompt_log.{key}: must be an integer >= {minimum}")

    if "max_file_bytes" in section:
        value = section["max_file_bytes"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append("prompt_log.max_file_bytes: must be an integer >= 0")

    return problems


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def extract_prompt(
    body_json: Any,
    *,
    endpoint: str = "chat/completions",
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> tuple[str, bool]:
    """Extract the prompt text for *body_json* and cap it.

    Chat completions use the first **non-system** message (delegating to
    :func:`proxy.router_helpers._first_message_content`, the shared extraction
    behind the existing body preview); ``/v1/completions`` uses the ``prompt``
    field (the first element when a list of prompts is supplied).

    Returns:
        ``(prompt_text, truncated)`` where *prompt_text* is capped at
        *max_prompt_chars*.
    """
    text = ""
    if isinstance(body_json, Mapping):
        if endpoint == "completions":
            raw = body_json.get("prompt")
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, list):
                text = next((item for item in raw if isinstance(item, str)), "")
        else:
            try:
                from proxy.router_helpers import _first_message_content

                text = _first_message_content(body_json)
            except Exception:  # pragma: no cover - defensive
                text = ""

    try:
        limit = int(max_prompt_chars)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        limit = DEFAULT_MAX_PROMPT_CHARS
    if limit < 1:
        limit = 1

    if len(text) > limit:
        return text[:limit], True
    return text, False


def _error_marker_from_response(response: Any) -> str | None:
    """Return ``error:<code>`` for a 4xx/5xx *response*, else ``None``."""
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or status < 400:
        return None

    code: str | None = None
    body = getattr(response, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            payload = json.loads(bytes(body))
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                candidate = error.get("code") or error.get("type")
                if isinstance(candidate, str) and candidate:
                    code = candidate
        except (ValueError, TypeError):
            code = None

    return f"error:{code}" if code else f"error:http_{status}"


def _error_marker_from_exception(exc: BaseException) -> str:
    """Return ``error:<kind>`` for an exception raised while dispatching."""
    status = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "detail", "") or "")
    if status == 400 and detail.startswith("Unknown model"):
        return "error:unknown_model"
    if isinstance(status, int):
        return f"error:http_{status}"
    return "error:exception"


def resolve_routed_model(*, response: Any = None, exc: BaseException | None = None) -> str:
    """Resolve the model that actually served the request (AC2).

    Priority: ``X-Resolved-Model`` (``local/<llama_model>`` or
    ``<provider>/<model>``) → ``X-Provider`` → failure marker → ``"unknown"``.
    """
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        try:
            resolved = headers.get("X-Resolved-Model")
        except Exception:  # pragma: no cover - defensive
            resolved = None
        if resolved:
            return str(resolved)

        marker = _error_marker_from_response(response)
        if marker:
            return marker

        try:
            provider = headers.get("X-Provider")
        except Exception:  # pragma: no cover - defensive
            provider = None
        if provider:
            return str(provider)
        return "unknown"

    if exc is not None:
        return _error_marker_from_exception(exc)

    return "unknown"


def build_entry(
    *,
    session_id: str,
    endpoint: str,
    requested_model: str | None,
    routed_model: str,
    prompt: str,
    truncated: bool,
    status_code: int | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one prompt-log entry (the JSONL line's object)."""
    return {
        "timestamp": timestamp or utc_timestamp(),
        "session_id": session_id,
        "endpoint": endpoint,
        "requested_model": requested_model or "unknown",
        "routed_model": routed_model,
        "status_code": status_code,
        "prompt": prompt,
        "truncated": bool(truncated),
    }


def _resolve_session_id(headers: Mapping[str, Any] | None) -> str:
    """Resolve the session ID via the shared header chain, else ``"unknown"``."""
    try:
        from proxy.session import _resolve_session_id_header

        session_id, _source = _resolve_session_id_header(headers or {})
    except Exception:  # pragma: no cover - defensive
        session_id = None
    return str(session_id) if session_id else "unknown"


def _parse_body(body: bytes | bytearray | str | None) -> Any:
    """Best-effort JSON parse of a request body."""
    if not body:
        return None
    try:
        if isinstance(body, (bytes, bytearray)):
            return json.loads(bytes(body).decode("utf-8", errors="replace"))
        return json.loads(body)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class _FlushRequest:
    """Sentinel queued by :meth:`PromptLog.flush`."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class PromptLog:
    """Append-only JSONL writer with a bounded tail reader.

    The writer thread is a daemon: it never keeps the process alive and a
    failed flush cannot hang shutdown. All file I/O happens off the request
    coroutine (``record`` only enqueues).
    """

    def __init__(self, config: PromptLogConfig, *, log: logging.Logger | None = None) -> None:
        self._config = config
        self._logger = log or logger
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._file_lock = threading.Lock()
        self._counters = {
            "recorded": 0,
            "write_errors": 0,
            "dropped": 0,
            "truncated": 0,
        }
        self._counters_lock = threading.Lock()
        self._last_error_log = 0.0
        self._estimated_size: int | None = None
        self._thread: threading.Thread | None = None
        if config.enabled:
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="prompt-log-writer",
                daemon=True,
            )
            self._thread.start()

    # -- properties -----------------------------------------------------

    @property
    def config(self) -> PromptLogConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def path(self) -> Path:
        return self._config.path

    # -- public API -----------------------------------------------------

    def record(self, entry: Mapping[str, Any]) -> bool:
        """Enqueue *entry* for append. Never raises; returns ``False`` when dropped."""
        if not self._config.enabled:
            return False
        try:
            line = json.dumps(dict(entry), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            self._bump("write_errors")
            self._log_error("prompt log entry not JSON-serialisable", exc_info=True)
            return False

        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._bump("dropped")
            self._log_error("prompt log queue full; entry dropped")
            return False

        self._bump("recorded")
        if entry.get("truncated"):
            self._bump("truncated")
        return True

    def counters(self) -> dict[str, int]:
        """Return a copy of the append counters."""
        with self._counters_lock:
            return dict(self._counters)

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until the queue is drained (test/shutdown helper).

        Returns ``False`` when the writer thread is absent or did not drain
        within *timeout*; never raises.
        """
        if self._thread is None or not self._thread.is_alive():
            return self._queue.empty()
        request = _FlushRequest()
        try:
            self._queue.put(request, timeout=timeout)
        except queue.Full:
            return False
        return request.event.wait(timeout)

    def recent_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return up to *limit* most recent entries, newest first.

        Reads a bounded tail of the file (never the whole file) so a long
        running log cannot stall the UI endpoint. When the live file holds
        fewer than *limit* entries (just after a rotation), the newest rotated
        backups supply the remainder — at most ``max_rotated_files`` extra
        bounded tail reads.
        """
        cap = self._config.max_entries
        if limit is not None:
            try:
                cap = min(int(limit), cap)
            except (TypeError, ValueError):
                pass
        base = str(self._config.path)
        candidates = [self._config.path] + [
            Path(f"{base}.{index}")
            for index in range(1, self._config.max_rotated_files + 1)
        ]
        entries: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(entries) >= cap:
                break
            entries.extend(read_recent_entries(candidate, cap - len(entries)))
        return entries

    def close(self, timeout: float = 5.0) -> None:
        """Drain and stop the writer thread (idempotent; test helper)."""
        thread = self._thread
        if thread is None:
            return
        self.flush(timeout=timeout)
        self._thread = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - defensive
            return
        thread.join(timeout=timeout)

    # -- writer thread --------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _FlushRequest):
                    item.event.set()
                    continue
                if item is None:
                    return
                self._append_line(item)
            except Exception:  # pragma: no cover - _append_line logs its own
                self._bump("write_errors")
                self._log_error("prompt log append failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _append_line(self, line: str) -> None:
        path = self._config.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(path, len(line) + 1)
            with self._file_lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            self._estimated_size = (self._estimated_size or 0) + len(line) + 1
        except OSError:
            self._bump("write_errors")
            self._log_error(f"prompt log append failed for {path}", exc_info=True)

    def _rotate_if_needed(self, path: Path, incoming_bytes: int) -> None:
        limit = self._config.max_file_bytes
        if limit <= 0:
            return
        if self._estimated_size is None:
            try:
                self._estimated_size = path.stat().st_size
            except OSError:
                self._estimated_size = 0
        if self._estimated_size + incoming_bytes <= limit:
            return
        self._rotate_now(path, self._config.max_rotated_files)
        self._estimated_size = 0

    @staticmethod
    def _rotate_now(path: Path, max_files: int) -> None:
        """Rotate *path* to ``.1``, shifting older backups down (bounded).

        ``path.N`` is dropped, ``path.N-1`` -> ``path.N`` … ``path.1`` ->
        ``path.2``, then ``path`` -> ``path.1``. Total on-disk growth is thus
        bounded by ``(1 + max_files) * max_file_bytes``.
        """
        base = str(path)
        for index in range(max_files, 0, -1):
            source = Path(f"{base}.{index}")
            if not source.exists():
                continue
            try:
                if index >= max_files:
                    source.unlink()
                else:
                    os.replace(source, Path(f"{base}.{index + 1}"))
            except OSError:
                pass
        try:
            os.replace(path, Path(f"{base}.1"))
        except OSError:
            pass

    # -- internals ------------------------------------------------------

    def _bump(self, key: str) -> None:
        with self._counters_lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def _log_error(self, message: str, *, exc_info: bool = False) -> None:
        """Log an append problem at most once per :data:`ERROR_LOG_INTERVAL_SECONDS`."""
        now = time.monotonic()
        if now - self._last_error_log < ERROR_LOG_INTERVAL_SECONDS:
            return
        self._last_error_log = now
        self._logger.warning(
            "prompt_log: %s (errors=%s dropped=%s)",
            message,
            self.counters().get("write_errors", 0),
            self.counters().get("dropped", 0),
            exc_info=exc_info,
        )


# ---------------------------------------------------------------------------
# Bounded tail reader
# ---------------------------------------------------------------------------


def read_recent_entries(
    path: Path | str,
    limit: int,
    *,
    chunk_bytes: int = TAIL_CHUNK_BYTES,
    max_bytes: int = MAX_TAIL_BYTES,
) -> list[dict[str, Any]]:
    """Read up to *limit* newest entries from the JSONL file, newest first.

    Missing files, unreadable files and malformed lines yield ``[]`` / are
    skipped — never an exception. Only a bounded tail is read: at most
    *max_bytes* from the end of the file.
    """
    if limit <= 0:
        return []
    file_path = Path(path)
    try:
        if not file_path.is_file():
            return []
        with file_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            data = b""
            read_total = 0
            while position > 0 and data.count(b"\n") <= limit and read_total < max_bytes:
                step = min(chunk_bytes, position, max_bytes - read_total)
                if step <= 0:
                    break
                position -= step
                handle.seek(position)
                data = handle.read(step) + data
                read_total += step
    except OSError:
        return []

    lines = data.split(b"\n")
    if position > 0:
        # The first line is very likely partial — drop it.
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]

    entries: list[dict[str, Any]] = []
    for raw in reversed(lines):
        if len(entries) >= limit:
            break
        try:
            entry = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, TypeError):
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Module-level singleton + request hook
# ---------------------------------------------------------------------------


_instance: PromptLog | None = None
_instance_lock = threading.Lock()


def configure(config: Mapping[str, Any] | None, log_dir: Any = None) -> PromptLog:
    """Resolve the config and install the process-wide :class:`PromptLog`."""
    resolved = resolve_config(config, log_dir)
    install(PromptLog(resolved))
    return _instance  # type: ignore[return-value]


def install(instance: PromptLog | None) -> None:
    """Install *instance* as the process-wide prompt log (``None`` disables)."""
    global _instance
    with _instance_lock:
        previous = _instance
        _instance = instance
    if previous is not None and previous is not instance:
        previous.close(timeout=1.0)


def get_prompt_log() -> PromptLog | None:
    """Return the process-wide prompt log, or ``None`` when not configured."""
    return _instance


def reset() -> None:
    """Remove the process-wide prompt log without draining it (test helper)."""
    global _instance
    with _instance_lock:
        _instance = None


def record_exchange(
    *,
    headers: Mapping[str, Any] | None,
    path: str,
    body: bytes | bytearray | str | None,
    response: Any = None,
    exc: BaseException | None = None,
) -> dict[str, Any] | None:
    """Record one handled request in the prompt log (AC1/AC2).

    Best-effort and never fatal: returns the entry when it was enqueued,
    ``None`` when logging is disabled, not configured, irrelevant (not a
    recorded path) or failed. Anything unexpected is swallowed — a logging
    problem must never fail a request (AC3).
    """
    try:
        instance = _instance
        if instance is None or not instance.enabled:
            return None
        if path not in RECORDED_PATHS:
            return None

        body_json = _parse_body(body)
        requested_model = None
        if isinstance(body_json, Mapping):
            requested_model = body_json.get("model")
            if not isinstance(requested_model, str) or not requested_model:
                requested_model = None

        prompt, truncated = extract_prompt(
            body_json,
            endpoint=path,
            max_prompt_chars=instance.config.max_prompt_chars,
        )
        entry = build_entry(
            session_id=_resolve_session_id(headers),
            endpoint=f"v1/{path}",
            requested_model=requested_model,
            routed_model=resolve_routed_model(response=response, exc=exc),
            prompt=prompt,
            truncated=truncated,
            status_code=getattr(response, "status_code", None),
        )
        return entry if instance.record(entry) else None
    except Exception:  # pragma: no cover - defensive: never fail a request
        return None


def recent_payload(limit: int | None = None) -> dict[str, Any]:
    """Build the ``GET /logs/prompt-log`` payload.

    Shape::

        {"enabled": bool, "entries": [...newest first...], "counters": {...},
         "max_entries": int, "file": "prompt-log.jsonl"}

    A missing/empty log file yields ``entries: []`` (the UI renders its
    empty state); read failures never raise.
    """
    instance = _instance
    if instance is None:
        return {
            "enabled": False,
            "entries": [],
            "counters": {},
            "max_entries": DEFAULT_MAX_ENTRIES,
            "file": FILENAME,
        }

    payload: dict[str, Any] = {
        "enabled": instance.enabled,
        "entries": [],
        "counters": instance.counters(),
        "max_entries": instance.config.max_entries,
        "file": Path(instance.path).name,
    }
    if instance.enabled:
        try:
            payload["entries"] = instance.recent_entries(limit)
        except Exception:  # pragma: no cover - defensive
            payload["entries"] = []
    return payload
