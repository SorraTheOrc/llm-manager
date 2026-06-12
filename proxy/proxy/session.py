"""
Session Coordination Module

Session routing orchestration, delta/fallback/single-flight/slot coordination,
session restore signal detection, and content streaming output.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Session observability state (initialized on import, but tests should
# monkeypatch via server module for isolation)
# ===================================================================

session_restore_observability: dict = {
    "restore_success_total": 0,
    "restore_fallback_total": {},
    "delta_payload_bytes_total": 0,
}

session_single_flight_observability: dict = {
    "queue_events_total": 0,
    "reject_events_total": 0,
    "active_sessions_current": 0,
    "queue_depth_current": 0,
}

session_guardrail_observability: dict = {
    "guardrail_cutoff_total": 0,
    "guardrail_cutoff_reasons": {},
    "session_invalidation_total": 0,
    "session_invalidation_reasons": {},
}


# ===================================================================
# Observability recorders (access state via srv for test monkeypatch compat)
# ===================================================================

def _record_restore_success() -> None:
    srv = _srv()
    try:
        srv.session_restore_observability["restore_success_total"] = (
            int(srv.session_restore_observability.get("restore_success_total", 0))
            + 1
        )
    except Exception:
        pass


def _record_restore_fallback(reason: str) -> None:
    if not reason:
        return
    srv = _srv()
    try:
        bucket = srv.session_restore_observability.setdefault(
            "restore_fallback_total", {}
        )
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


def _record_delta_payload_bytes(value: int) -> None:
    if value <= 0:
        return
    srv = _srv()
    try:
        srv.session_restore_observability["delta_payload_bytes_total"] = (
            int(srv.session_restore_observability.get("delta_payload_bytes_total", 0))
            + int(value)
        )
    except Exception:
        pass


def _record_single_flight_queue() -> None:
    srv = _srv()
    try:
        srv.session_single_flight_observability["queue_events_total"] = (
            int(srv.session_single_flight_observability.get("queue_events_total", 0))
            + 1
        )
    except Exception:
        pass


def _record_single_flight_reject() -> None:
    srv = _srv()
    try:
        srv.session_single_flight_observability["reject_events_total"] = (
            int(
                srv.session_single_flight_observability.get(
                    "reject_events_total", 0
                )
            )
            + 1
        )
    except Exception:
        pass


def _record_guardrail_cutoff(reason: str) -> None:
    if not reason:
        return
    srv = _srv()
    try:
        srv.session_guardrail_observability["guardrail_cutoff_total"] = (
            int(
                srv.session_guardrail_observability.get(
                    "guardrail_cutoff_total", 0
                )
            )
            + 1
        )
        bucket = srv.session_guardrail_observability.setdefault(
            "guardrail_cutoff_reasons", {}
        )
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


def _record_session_invalidation(reason: str) -> None:
    if not reason:
        return
    srv = _srv()
    try:
        srv.session_guardrail_observability["session_invalidation_total"] = (
            int(
                srv.session_guardrail_observability.get(
                    "session_invalidation_total", 0
                )
            )
            + 1
        )
        bucket = srv.session_guardrail_observability.setdefault(
            "session_invalidation_reasons", {}
        )
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    except Exception:
        pass


# ===================================================================
# Session restore signal detection
# ===================================================================

def _detect_restore_signal_from_log_slice(
    log_path: Path,
    start_offset: int,
) -> bool:
    """Return True when restore evidence exists in newly appended log bytes."""
    if not log_path.exists():
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, int(start_offset)))
            data = f.read()
    except Exception:
        return False

    if not data:
        return False

    text = data.lower()
    phrases = (
        "restored context checkpoint",
        "load_session",
        "session restore",
        "restore session",
        "loading kv cache",
        "kv cache restored",
    )
    return any(p in text for p in phrases)


def _detect_restore_signal_from_llama_log(
    session_id: Optional[str],
    log_path: Optional[Path] = None,
    lookback_lines: int = 400,
) -> bool:
    """Best-effort compatibility signal from llama-server logs.

    Prefer session-id-specific lines when available.
    """
    srv = _srv()

    if not session_id:
        return False

    target_path = log_path or srv._resolve_log_path("llama")
    if not target_path.exists():
        return False

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-max(1, int(lookback_lines)):]
    except Exception:
        return False

    sid = str(session_id).strip()
    sid_lower = sid.lower()
    phrases = (
        "load_session",
        "session restore",
        "restore session",
        "loading kv cache",
        "kv cache restored",
        "restored context checkpoint",
    )

    for line in reversed(lines):
        text = line.strip().lower()
        if sid_lower in text and any(p in text for p in phrases):
            return True

    # Fallback: if no session id appears in log format, accept recent
    # restore phrases.
    return any(
        any(p in line.strip().lower() for p in phrases) for line in lines
    )


# ===================================================================
# Log path resolution
# ===================================================================

def _resolve_log_path(source: str = "proxy") -> Path:
    """Resolve the log file path for a given source.

    Args:
        source: Either 'proxy' for proxy.log or 'llama' for llama-server.log

    Returns:
        Path to the requested log file
    """
    srv = _srv()
    if source == "llama":
        if srv.log_dir:
            return srv.log_dir / "llama-server.log"
        else:
            return Path(__file__).parent / "logs" / "llama-server.log"
    else:
        if srv.log_dir:
            return srv.log_dir / "proxy.log"
        else:
            return Path(__file__).parent / "logs" / "proxy.log"


# ===================================================================
# Content extraction from SSE chunks
# ===================================================================

def extract_streamed_content_from_chunk(chunk_str: str) -> Optional[str]:
    """Extract concatenated delta.content and delta.reasoning_content strings
    from an SSE chunk.

    Returns the concatenated content (may include newlines as provided by the
    delta values) or None if no parseable content is found.
    Handles both 'content' and 'reasoning_content' fields used by models like
    Qwen3 during their thinking/reasoning phase.
    """
    if not chunk_str:
        return None
    try:
        contents: list[str] = []
        # First attempt: parse lines prefixed with 'data:' (SSE style)
        for line in chunk_str.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                try:
                    j = json.loads(payload)
                except Exception:
                    continue
                if isinstance(j, dict):
                    choices = j.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        if isinstance(delta, dict):
                            for key in ("reasoning_content", "content"):
                                content_piece = delta.get(key)
                                if content_piece is not None:
                                    contents.append(str(content_piece))
        if contents:
            return "".join(contents)

        # Second attempt: try to parse the whole chunk as JSON
        s = chunk_str.strip()
        if s:
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    choices = j.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        if isinstance(delta, dict):
                            for key in ("reasoning_content", "content"):
                                content_piece = delta.get(key)
                                if content_piece is not None:
                                    contents.append(str(content_piece))
                if contents:
                    return "".join(contents)
            except Exception:
                pass
    except Exception:
        pass
    return None


# ===================================================================
# Content-only console handler
# ===================================================================

class ContentOnlyConsoleHandler(logging.StreamHandler):
    """Console handler that prints only streamed content for STREAM CHUNK
    records.

    For log records whose formatted message begins with the prefix
    "STREAM CHUNK | ", this handler will attempt to extract delta.content
    values from any JSON payloads inside the chunk and write only the
    concatenated content to the console stream (without adding extra
    newlines). Raw JSON is never displayed in the console — only extracted
    text content is shown.

    - reasoning_content is displayed in dim/grey
    - content is displayed in bold

    For other records, normal formatting is used.
    """

    PREFIX = "STREAM CHUNK | "
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def _extract_and_format_content(self, chunk_str: str) -> Optional[str]:
        """Extract content from chunk and apply formatting based on type.

        Returns formatted string with ANSI codes, or None if no content
        found.
        """
        if not chunk_str:
            return None

        parts: list[str] = []
        for line in chunk_str.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            for choice in j.get("choices", []):
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    continue
                reasoning = delta.get("reasoning_content")
                if reasoning is not None:
                    parts.append(f"{self.DIM}{reasoning}{self.RESET}")
                content = delta.get("content")
                if content is not None:
                    parts.append(f"{self.BOLD}{content}{self.RESET}")

        return "".join(parts) if parts else None

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = record.getMessage()
            if isinstance(msg, str) and msg.startswith(self.PREFIX):
                chunk_str = msg[len(self.PREFIX):]
                formatted = self._extract_and_format_content(chunk_str)
                if formatted:
                    if getattr(self, "stream", None) is None:
                        self.stream = sys.stderr
                    try:
                        self.stream.write(formatted)
                        try:
                            self.flush()
                        except Exception:
                            pass
                    except Exception:
                        pass
                return
            super().emit(record)
        except Exception:
            try:
                super().emit(record)
            except Exception:
                pass
