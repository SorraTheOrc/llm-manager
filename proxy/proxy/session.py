"""
Session Coordination Module

Session routing orchestration, delta/fallback/single-flight/slot coordination,
session restore signal detection, and content streaming output.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
        with open(log_path, encoding="utf-8", errors="replace") as f:
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
    session_id: str | None,
    log_path: Path | None = None,
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
        with open(target_path, encoding="utf-8", errors="replace") as f:
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


def extract_streamed_content_from_chunk(chunk_str: str) -> str | None:
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


def extract_streamed_assistant_message_from_sse(
    sse_text: str,
) -> dict[str, Any] | None:
    """Reconstruct an assistant message from streaming SSE text.

    Collects ``content``, ``reasoning_content``, and streamed ``tool_calls``
    deltas so the persisted session history retains the same token-bearing
    fields that were present in the upstream response.
    """
    if not sse_text:
        return None

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    def _merge_tool_call(target: dict[str, Any], delta: dict[str, Any]) -> None:
        if not isinstance(delta, dict):
            return
        if delta.get("id") and not target.get("id"):
            target["id"] = str(delta["id"])
        if delta.get("type") and not target.get("type"):
            target["type"] = str(delta["type"])
        function_delta = delta.get("function")
        if isinstance(function_delta, dict):
            function_target = target.setdefault("function", {})
            name = function_delta.get("name")
            if isinstance(name, str) and name and not function_target.get("name"):
                function_target["name"] = name
            arguments = function_delta.get("arguments")
            if arguments is not None:
                function_target["arguments"] = str(function_target.get("arguments", "")) + str(arguments)

    try:
        for line in sse_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            for choice in j.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if content is not None:
                    content_parts.append(str(content))
                reasoning = delta.get("reasoning_content")
                if reasoning is not None:
                    reasoning_parts.append(str(reasoning))
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for fallback_index, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, dict):
                            continue
                        raw_index = tool_call.get("index", fallback_index)
                        try:
                            tool_index = int(raw_index)
                        except Exception:
                            tool_index = fallback_index
                        target = tool_calls_by_index.setdefault(
                            tool_index,
                            {"function": {}},
                        )
                        _merge_tool_call(target, tool_call)
    except Exception:
        return None

    if not (content_parts or reasoning_parts or tool_calls_by_index):
        return None

    assistant_message: dict[str, Any] = {"role": "assistant"}
    if content_parts:
        assistant_message["content"] = "".join(content_parts)
    if reasoning_parts:
        assistant_message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls_by_index:
        assistant_message["tool_calls"] = [
            tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
        ]
    return assistant_message


# ===================================================================
# Content-only console handler
# ===================================================================

class ContentOnlyConsoleHandler(logging.StreamHandler):
    """Console handler for streaming-related log records.

    For log records whose formatted message begins with the prefix
    "STREAM CHUNK | ", this handler **suppresses** console output entirely
    (the raw JSON continues to be logged to file handlers only).

    For other records (e.g., lifecycle log lines like ``Stream started:``,
    ``Stream finished:``, ``Stream error:``), normal formatting is used.

    This is intentionally a no-op for STREAM CHUNK records to prevent
    interleaved content in the console when multiple streaming responses
    are active simultaneously (LP-0MR90HJED005WI1Z).
    """

    PREFIX = "STREAM CHUNK | "

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = record.getMessage()
            if isinstance(msg, str) and msg.startswith(self.PREFIX):
                # Suppress STREAM CHUNK content from console output.
                # The raw JSON continues to be written to file handlers
                # independently (LP-0MR90HJED005WI1Z).
                return
            super().emit(record)
        except Exception:
            try:
                super().emit(record)
            except Exception:
                pass


# ===================================================================
# Slot directory and persistence helpers
# ===================================================================


def _ensure_slot_dir(slot_path: str | None) -> Path | None:
    if not slot_path:
        return None
    try:
        path = Path(slot_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _slot_persistence_enabled(slot_path: Path | str | None, slot_pool_size: int) -> bool:
    return bool(slot_path and slot_pool_size > 0)


# ---------------------------------------------------------------------------
# Slot persistence safeguards (LP-0MS91DHPZ001VWQO)
#
# llama-server serializes the whole KV cache on slot save/restore, so large
# contexts can take longer than the client timeout (ReadTimeout). A copy that
# never completes leaves the GPU busy with idle slots, degrading all local
# inference. These safeguards are proxy-side only (no llama.cpp changes):
#
#   1. Context-size gating: skip persistence when the request context exceeds
#      ``session_slot_max_prompt_tokens``.
#   2. Adaptive timeout: scale the save/restore timeout with context size
#      (``session_slot_timeout_per_token_seconds``, capped at
#      ``session_slot_max_timeout_seconds``).
#   3. Circuit breaker: after ``session_slot_max_consecutive_failures``
#      failures, stop issuing save/restore for a slot until the cooldown
#      (``session_slot_failure_cooldown_seconds``) expires.
# ---------------------------------------------------------------------------

_slot_failure_state: dict[int, tuple[int, float]] = {}
"""{slot_id: (consecutive_failures, last_failure_epoch)} for the circuit breaker."""


def _record_slot_success(slot_id: int) -> None:
    """Reset the failure state for *slot_id* after a successful save/restore."""
    _slot_failure_state.pop(slot_id, None)


def _record_slot_failure(slot_id: int) -> None:
    """Increment the consecutive-failure counter for *slot_id*."""
    count, _ = _slot_failure_state.get(slot_id, (0, time.time()))
    _slot_failure_state[slot_id] = (count + 1, time.time())


def _estimate_slot_prompt_tokens(
    body_json: dict | None,
    server_config: dict | None = None,
    model_config: dict | None = None,
) -> int:
    """Estimate request context tokens for slot persistence decisions.

    Uses the SAME shared tokenizer resolution chain as the routing estimate
    (``_get_tokenizer_for_model`` in provider.py, LP-0MSEQ71IF0003FRT): a
    named tokenizer on the model entry gives exact native counts (multiplier
    forced to 1.0), otherwise tiktoken + the server-level
    ``token_estimate_multiplier`` applies. This guarantees the persistence
    cap compares Qwen3-native token counts consistently with the routing
    clamp (LP-0MSEGPO77005CYCQ F2, AC3).

    Args:
        body_json: Parsed request body.
        server_config: Server config section (may carry
            ``token_estimate_multiplier`` for the fallback path).
        model_config: Optional model entry config (may carry
            ``tokenizer: <name>``). When None, resolves to the
            server-level multiplier fallback (no native tokenizer).
    """
    if not isinstance(body_json, dict):
        return 0
    from proxy.provider import (
        _estimate_prompt_tokens_for_routing,
        _get_tokenizer_for_model,
    )

    tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config or {})
    estimate = _estimate_prompt_tokens_for_routing(body_json, tokenizer=tokenizer) or 0
    if multiplier != 1.0:
        estimate = int(estimate * multiplier)
    return estimate


def _slot_in_failure_cooldown(slot_id: int, server_config: dict) -> bool:
    """True when a slot has exceeded the failure threshold within the cooldown window."""
    failures, last_fail = _slot_failure_state.get(slot_id, (0, 0.0))
    max_failures = int(server_config.get("session_slot_max_consecutive_failures", 3) or 3)
    cooldown = float(server_config.get("session_slot_failure_cooldown_seconds", 300.0) or 300.0)
    if failures < max_failures:
        return False
    remaining = cooldown - (time.time() - last_fail)
    if remaining > 0:
        logger.warning(
            "slot persistence disabled slot=%s consecutive_failures=%s cooldown_remaining=%.0fs",
            slot_id, failures, remaining,
        )
        return True
    # Cooldown expired - allow retries again
    _slot_failure_state.pop(slot_id, None)
    return False


def _truncate_body(body: str, maxlen: int = 500) -> str:
    """Truncate *body* to *maxlen* characters, appending '...' if truncated.

    Used when logging HTTP response bodies to prevent sensitive data
    (conversation content, model responses) from flooding logs.
    """
    if len(body) <= maxlen:
        return body
    return body[:maxlen] + "..."


def _slot_busy_state_snapshot(
    slot_id: int | None = None,
    exclude_session: str | None = None,
) -> dict:
    """Snapshot proxy-side load state used to judge slot busyness.

    Returns a JSON-serializable dict with:

    - ``active_queries``: total in-flight queries (local + remote backends).
    - ``local_active_queries``: in-flight queries on the local backend.
    - ``active_sessions``: sessions holding an *active* dispatch lease
      (a request currently in flight on the local backend), excluding
      *exclude_session* when provided.
    - ``slot_busy``: whether *slot_id*'s owning session has an active
      dispatch lease (a request in flight on that slot).

    Correlation evidence (LP-0MSI1RWLM007N367 F1): slot save/restore
    ReadTimeouts cluster when other local streams are active — llama-server
    serializes each KV-cache copy behind its slot work, so a save issued
    while another slot is generating can exceed the adaptive timeout. This
    snapshot gives the correlation script (and the F3 load gate) a concrete
    busy definition from proxy-side state, with no extra llama-server calls.

    *exclude_session* is used by the F3 load gate so the requesting
    session's own active lease does not make the slot look busy (the gate
    must reflect *other* sessions' load).
    """
    state = {
        "active_queries": 0,
        "local_active_queries": 0,
        "active_sessions": 0,
        "slot_busy": False,
    }
    try:
        srv = _srv()
    except Exception:
        return state
    try:
        state["active_queries"] = int(
            getattr(srv, "active_queries", 0) or 0
        )
    except Exception:
        pass
    try:
        state["local_active_queries"] = int(
            getattr(srv, "local_active_queries", 0) or 0
        )
    except Exception:
        pass
    try:
        records = getattr(srv, "local_dispatch_records", None)
        if isinstance(records, dict):
            state["active_sessions"] = sum(
                1
                for owner, rec in records.items()
                if isinstance(rec, dict)
                and rec.get("active")
                and owner != exclude_session
            )
            if slot_id is not None:
                owner = None
                for sid, owner_id in _slot_owners.items():
                    if sid == slot_id:
                        owner = owner_id
                        break
                if owner is not None and owner in records:
                    rec = records.get(owner)
                    state["slot_busy"] = bool(
                        isinstance(rec, dict) and rec.get("active")
                    )
    except Exception:
        pass
    return state


def _slot_persistence_skip_when_busy(
    server_config: dict,
    slot_id: int | None,
    session_id: str | None = None,
) -> bool:
    """Return True when the load-aware gate says persistence should be skipped.

    Implements the F3 busy-signal decision (LP-0MSI1RWLM007N367 F3, per the
    F1 decision record): skip slot save/restore when another local session is
    actively streaming — llama-server serializes each KV-cache copy behind
    its slot work, so saves issued under concurrent load exceed the adaptive
    timeout. The signal is proxy-side dispatch-lease state (no llama-server
    calls); the requesting session's own lease is excluded.

    Disabled when ``session_slot_skip_when_busy`` is false. The gate is
    enabled explicitly by the deployed config (``proxy/config.yaml`` sets
    ``session_slot_skip_when_busy: true``); a non-dict *server_config* or an
    absent key is treated as gate-off so `_build_slot_context` does not
    depend on ambient global server state unless the feature is opted in.
    """
    if not isinstance(server_config, dict):
        return False
    if not server_config.get("session_slot_skip_when_busy", False):
        return False
    busy = _slot_busy_state_snapshot(slot_id, exclude_session=session_id)
    return busy["active_sessions"] > 0


async def _call_slot_endpoint(
    llama_port: int,
    slot_id: int,
    action: str,
    filename: str,
    timeout: float,
    model: str | None = None,
) -> bool:
    """Make a slot save/restore HTTP call to llama-server.

    Improved logging (LP-0MQWXX17C005BX1E):
    - Exceptions include the exception type name in the warning message so
      empty __str__ values never produce empty error fields.
    - Non-200 responses are logged at WARNING with status code and a truncated
      (≤500 char) response body.
    - A DEBUG-level log with exc_info=True is emitted for every exception,
      capturing the full stack trace for post-hoc diagnosis.

    Failure-path instrumentation (LP-0MSI1RWLM007N367 F1): every failed
    save/restore logs the elapsed time at timeout, the applied timeout, and a
    proxy-side busy-state snapshot (``_slot_busy_state_snapshot``) so the
    correlation script can prove whether the failure was load-starved. The
    success path logs elapsed time at DEBUG for latency measurement.

    Failures feed the circuit breaker (``_slot_failure_state``) so repeated
    timeouts disable persistence for the slot instead of piling doomed
    KV-cache copies onto the GPU (LP-0MS91DHPZ001VWQO).
    """
    if not filename:
        return False
    url = f"http://localhost:{llama_port}/slots/{slot_id}?action={action}"
    payload = {"filename": Path(filename).name}
    if model:
        payload["model"] = model
    srv = _srv()
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=timeout)
    start = time.monotonic()
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        elapsed = time.monotonic() - start
        if getattr(response, "status_code", None) != 200:
            body = getattr(response, "text", "")
            busy = _slot_busy_state_snapshot(slot_id)
            srv.logger.warning(
                "slot_%s failed slot=%s status=%s body=%s elapsed=%.1fs timeout=%.1fs busy=%s",
                action,
                slot_id,
                response.status_code,
                _truncate_body(body),
                elapsed,
                timeout,
                json.dumps(busy),
            )
            _record_slot_failure(slot_id)
            return False
        _record_slot_success(slot_id)
        # Elapsed-time on success lets the correlation script measure real
        # save/restore latency vs the adaptive timeout (llama-server logs
        # lack timestamps; LP-0MSI1RWLM007N367 F1).
        srv.logger.debug(
            "slot_%s ok slot=%s elapsed=%.1fs timeout=%.1fs",
            action,
            slot_id,
            elapsed,
            timeout,
        )
        return True
    except Exception as exc:
        elapsed = time.monotonic() - start
        # Log exception type name in the warning so empty __str__ values
        # never produce an empty error field (LP-0MQWXX17C005BX1E).
        exc_type = type(exc).__name__
        detail = str(exc) if str(exc) else exc_type
        busy = _slot_busy_state_snapshot(slot_id)
        srv.logger.warning(
            "slot_%s failed slot=%s error=%s/%s elapsed=%.1fs timeout=%.1fs busy=%s",
            action,
            slot_id,
            exc_type,
            detail,
            elapsed,
            timeout,
            json.dumps(busy),
        )
        # Feed the circuit breaker so repeated timeouts disable persistence.
        _record_slot_failure(slot_id)
        # Debug log with full traceback for post-hoc diagnosis.
        srv.logger.debug(
            "slot_%s failed slot=%s",
            action,
            slot_id,
            exc_info=True,
        )
        return False
    finally:
        if not srv._http_client:
            try:
                await client.aclose()
            except Exception:
                pass


async def _restore_slot_snapshot(
    llama_port: int,
    slot_id: int,
    filename: str,
    timeout: float,
    model: str | None = None,
) -> bool:
    try:
        if not Path(filename).exists():
            return False
    except Exception:
        return False
    return await _call_slot_endpoint(
        llama_port,
        slot_id,
        "restore",
        filename,
        timeout,
        model=model,
    )


async def _save_slot_snapshot(
    llama_port: int,
    slot_id: int,
    filename: str,
    timeout: float,
    model: str | None = None,
) -> bool:
    return await _call_slot_endpoint(
        llama_port,
        slot_id,
        "save",
        filename,
        timeout,
        model=model,
    )


# ===================================================================
# Session ID helpers
# ===================================================================


def _sanitize_session_id(session_id: str) -> str:
    if not session_id:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)


# ---------------------------------------------------------------------------
# Slot assignment registry
#
# Tracks which session_id currently owns each slot number, replacing the
# previous hash-based deterministic mapping. Slots are assigned on demand
# (lowest-numbered free slot first) so that session_id ↔ slot_number is
# transparent and doesn't rely on a digest.
# ---------------------------------------------------------------------------

_slot_owners: dict[int, str] = {}
"""Active slot assignments: ``{slot_id: session_id}``."""


def _slot_id_for_session(session_id: str, pool_size: int) -> int | None:
    """Return the slot number assigned to *session_id*.

    If the session already has an assigned slot, returns it.
    Otherwise assigns the lowest-numbered free slot in ``[0, pool_size)``.
    Returns ``None`` when *session_id* is empty or *pool_size* ≤ 0.
    """
    if not session_id or pool_size <= 0:
        return None

    # Check existing assignment
    for sid, owner in list(_slot_owners.items()):
        if owner == session_id:
            return sid

    # Find the lowest-numbered free slot
    used = set(_slot_owners.keys())
    for sid in range(pool_size):
        if sid not in used:
            _slot_owners[sid] = session_id
            return sid

    # All slots occupied — pool is exhausted
    return None


def _free_slot_assignment(session_id: str) -> None:
    """Release the slot assigned to *session_id*, if any."""
    if not session_id:
        return
    for sid, owner in list(_slot_owners.items()):
        if owner == session_id:
            _slot_owners.pop(sid, None)
            return


def _assigned_slot_for_session(session_id: str) -> int | None:
    """Return the slot currently assigned to *session_id*, if any.

    Read-only lookup: unlike ``_slot_id_for_session`` it never assigns a
    new slot. Used by orphan-cleanup verification to check whether a
    session's slot is still processing on llama-server before freeing the
    dispatch lease (LP-0MSUO6XRP001MCB2). Returns ``None`` when the session
    has no slot assignment or *session_id* is empty.
    """
    if not session_id:
        return None
    for sid, owner in list(_slot_owners.items()):
        if owner == session_id:
            return sid
    return None


def _slot_filename_for_session(session_id: str, base_dir: Path | str) -> str:
    safe_id = _sanitize_session_id(session_id)
    return str(Path(base_dir) / f"slot_{safe_id}.bin")


def _build_slot_context(
    server_config: dict,
    session_id: str | None,
    body_json: dict | None = None,
) -> tuple[int | None, str | None, float]:
    """Build the slot context for a request, applying persistence safeguards.

    Returns ``(None, None, timeout)`` — disabling slot persistence for this
    request — when any of these apply (LP-0MS91DHPZ001VWQO):

    - the request context exceeds ``session_slot_max_prompt_tokens``
      (saves serialize the whole KV cache, so large contexts ReadTimeout
      and can wedge the GPU), or
    - another local session is actively streaming (load-aware gate,
      ``session_slot_skip_when_busy``; LP-0MSI1RWLM007N367 F3), or
    - the slot is in circuit-breaker cooldown after repeated save/restore
      failures.

    When ``session_slot_timeout_per_token_seconds`` is configured, the
    returned timeout scales with the estimated context size, capped at
    ``session_slot_max_timeout_seconds``.
    """
    slot_path = server_config.get("session_slot_save_path")
    slot_pool_size = int(server_config.get("session_slot_pool_size", 0) or 0)
    slot_timeout = float(server_config.get("session_slot_timeout_seconds", 3.0) or 3.0)
    slot_dir = _ensure_slot_dir(slot_path)
    if not session_id or not _slot_persistence_enabled(slot_dir, slot_pool_size):
        return None, None, slot_timeout
    slot_id = _slot_id_for_session(session_id, slot_pool_size)
    if slot_id is None:
        return None, None, slot_timeout

    # 1) Context-size gating: skip persistence for oversized contexts.
    max_prompt_tokens = int(server_config.get("session_slot_max_prompt_tokens", 0) or 0)
    # Dynamic derivation (LP-0MSEGPO77005CYCQ F3): when the static config
    # key is absent/0, derive the cap from the effective per-slot clamp — the
    # SAME source as the routing clamp (local_model_ctx_size // active_slots
    # - output headroom) — so it auto-adapts to slot-count/ctx-size changes
    # instead of freezing at a static value.
    # LP-0MTBOX45O005LD1S: when a hard-routing cap is configured, use that
    # as the single source for both persistence and routing so cache-restore
    # is never attempted above the routing cap (consistent AC4).
    if max_prompt_tokens <= 0:
        # Try the hard cap first (mode-aware); fall through to the
        # per-slot threshold when no hard cap is configured.
        try:
            from proxy.mode import read_mode as _read_mode
            from proxy.provider import compute_hard_routing_cap
            _mode = _read_mode()
            _hard_cap = compute_hard_routing_cap(_mode, server_config)
            if _hard_cap > 0:
                max_prompt_tokens = _hard_cap
        except Exception:
            pass
        # LP-0MTE9HAF8008909G F3: pin the dynamic persistence cap to the
        # per-slot routing clamp (83285 / 126976 for the live 262K schedule)
        # so the router and the persistence gate use the same threshold.
        # The previous hard-cap derivation (70000 / 61440) left a gap where
        # the router accepted a request but persistence gated it out, which
        # disabled KV reuse for the largest sessions that benefit most.
        from proxy.provider import (
            _get_active_local_ctx_size as _persist_ctx_size,
            _get_active_local_slots as _persist_slots,
            effective_per_slot_threshold as _persist_clamp,
        )
        _persist_ctx = _persist_ctx_size(server_config)
        _persist_n = _persist_slots(server_config)
        _persist_threshold = _persist_clamp(_persist_ctx, _persist_n)
        if _persist_threshold > max_prompt_tokens:
            max_prompt_tokens = _persist_threshold
    per_token = float(server_config.get("session_slot_timeout_per_token_seconds", 0.0) or 0.0)
    need_estimate = max_prompt_tokens > 0 or per_token > 0
    # Resolve the request's model config so the persistence estimate uses the
    # SAME tokenizer resolution chain as the routing clamp (AC3,
    # LP-0MSEQ71IF0003FRT). Falls back to None (server-level multiplier only)
    # when the model cannot be resolved (e.g. unit tests without a live
    # server, or requests with no model field).
    model_config = None
    if isinstance(body_json, dict) and body_json.get("model"):
        try:
            from proxy.lifecycle import get_model_config

            model_config = get_model_config(body_json.get("model"))
        except Exception:
            model_config = None
    estimated = (
        _estimate_slot_prompt_tokens(body_json, server_config, model_config)
        if need_estimate
        else 0
    )
    if max_prompt_tokens > 0 and estimated > max_prompt_tokens:
        logger.info(
            "slot persistence skipped session=%s estimated_tokens=%d max_prompt_tokens=%d reason=context_too_large",
            session_id[:8] if session_id else "unknown",
            estimated,
            max_prompt_tokens,
        )
        return None, None, slot_timeout

    # 2) Load-aware gating (LP-0MSI1RWLM007N367 F3): skip persistence when
    #    another local session is actively streaming. llama-server serializes
    #    each KV-cache copy behind its slot work, so saves issued under
    #    concurrent load exceed the adaptive timeout (ReadTimeout). The busy
    #    signal is proxy-side dispatch-lease state (no llama-server calls);
    #    the requesting session's own lease is excluded (at request start the
    #    session is not yet in the records anyway). Configurable via
    #    ``session_slot_skip_when_busy`` (enabled by the deployed config.yaml).
    if _slot_persistence_skip_when_busy(server_config, slot_id, session_id):
        busy = _slot_busy_state_snapshot(slot_id, exclude_session=session_id)
        logger.info(
            "slot persistence skipped session=%s slot=%s reason=slot_busy active_sessions=%s",
            session_id[:8] if session_id else "unknown",
            slot_id,
            busy["active_sessions"],
        )
        return None, None, slot_timeout

    # 3) Adaptive timeout: scale with context size so legitimate saves of
    #    medium contexts don't spuriously time out.
    if per_token > 0:
        max_timeout = float(server_config.get("session_slot_max_timeout_seconds", 60.0) or 60.0)
        slot_timeout = min(slot_timeout + per_token * estimated, max_timeout)

    # 4) Circuit breaker: stop issuing saves/restores for a failing slot.
    if _slot_in_failure_cooldown(slot_id, server_config):
        return None, None, slot_timeout

    return slot_id, _slot_filename_for_session(session_id, slot_dir), slot_timeout


async def _invalidate_session_and_slot(
    session_id: str | None,
    reason: str,
    slot_filename: str | None,
) -> None:
    """
    Invalidate a session and clean up its slot file.

    The dispatch lease system (``_try_acquire_local_dispatch``) handles
    concurrency gating, session ownership, and timeout-based release
    independently — no scheduler admission layer exists.

    Args:
        session_id: The session to invalidate.
        reason: Invalidation reason string.
        slot_filename: Path to the slot persistence file to remove.
    """
    if session_id:
        try:
            srv = _srv()
            await srv.session_manager.invalidate(session_id)
        except Exception:
            pass

        # Also release any dispatch lease record for this session
        try:
            srv = _srv()
            async with srv.local_dispatch_records_lock:
                if session_id in srv.local_dispatch_records:
                    del srv.local_dispatch_records[session_id]
                    try:
                        _srv().logger.info(
                            "lease_released session=%s reason=%s",
                            session_id if session_id else "unknown",
                            reason or "invalidation",
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    # Free the slot registry entry
    _free_slot_assignment(session_id)

    # Clear cached ratio entries for this session (LP-0MRMMBZ7T007ER59)
    if session_id:
        try:
            from proxy.provider import _last_cached_ratio
            keys_to_delete = [
                k for k in _last_cached_ratio if k[1] == session_id
            ]
            for k in keys_to_delete:
                _last_cached_ratio.pop(k, None)
        except Exception:
            pass

    if reason:
        _record_session_invalidation(reason)
    if slot_filename:
        try:
            Path(slot_filename).unlink(missing_ok=True)
        except Exception:
            pass


# ===================================================================
# Slot lock coordination
# ===================================================================


class SlotLockCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}
    def acquire(self, slot_id: int | None):
        @asynccontextmanager
        async def _guard():
            if slot_id is None:
                yield
                return
            async with self._lock:
                lock = self._locks.get(slot_id)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[slot_id] = lock
            await lock.acquire()
            try:
                yield
            finally:
                lock.release()

        return _guard()


slot_lock_coordinator = SlotLockCoordinator()


# ===================================================================
# Session single-flight coordination
# ===================================================================


class SessionSingleFlightRejectedError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SessionSingleFlightCoordinator:
    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._states: dict[str, dict[str, Any]] = {}

    async def _get_state(self, session_id: str) -> dict[str, Any]:
        async with self._state_lock:
            state = self._states.get(session_id)
            if state is None:
                state = {"lock": asyncio.Lock(), "waiters": 0, "active": False}
                self._states[session_id] = state
            return state

    def acquire(self, session_id: str | None, mode: str, max_queue_depth: int, queue_timeout_seconds: float | None = None):
        @asynccontextmanager
        async def _guard():
            if not session_id:
                yield
                return

            state = await self._get_state(session_id)
            mode_norm = (mode or "queue").strip().lower()
            if mode_norm not in {"queue", "reject"}:
                mode_norm = "queue"

            is_waiting = False
            async with self._state_lock:
                if state["lock"].locked():
                    if mode_norm == "reject":
                        _record_single_flight_reject()
                        raise SessionSingleFlightRejectedError("active_inflight")
                    if max_queue_depth is not None and state["waiters"] >= max_queue_depth:
                        _record_single_flight_reject()
                        raise SessionSingleFlightRejectedError("queue_full")
                    state["waiters"] += 1
                    is_waiting = True
                    _record_single_flight_queue()

            try:
                if queue_timeout_seconds is not None and queue_timeout_seconds > 0:
                    await asyncio.wait_for(
                        state["lock"].acquire(), timeout=queue_timeout_seconds
                    )
                else:
                    await state["lock"].acquire()
            except TimeoutError:
                async with self._state_lock:
                    if is_waiting:
                        state["waiters"] = max(0, state["waiters"] - 1)
                _record_single_flight_reject()
                raise SessionSingleFlightRejectedError("queue_timeout")
            async with self._state_lock:
                if is_waiting:
                    state["waiters"] = max(0, state["waiters"] - 1)
                state["active"] = True
                session_single_flight_observability["active_sessions_current"] = sum(
                    1 for s in self._states.values() if s.get("active")
                )
                session_single_flight_observability["queue_depth_current"] = sum(
                    int(s.get("waiters", 0)) for s in self._states.values()
                )

            try:
                yield
            finally:
                state["lock"].release()
                async with self._state_lock:
                    state["active"] = False
                    if not state["lock"].locked() and state["waiters"] == 0:
                        self._states.pop(session_id, None)
                    session_single_flight_observability["active_sessions_current"] = sum(
                        1 for s in self._states.values() if s.get("active")
                    )
                    session_single_flight_observability["queue_depth_current"] = sum(
                        int(s.get("waiters", 0)) for s in self._states.values()
                    )

        return _guard()

    def metrics_snapshot(self) -> dict:
        return dict(session_single_flight_observability)


session_single_flight_coordinator = SessionSingleFlightCoordinator()


# ===================================================================
# Guardrail / repetition detection
# ===================================================================


def _should_cutoff_for_repetition(
    response_text: str,
    min_pattern_chars: int,
    min_repeats: int,
) -> bool:
    if not response_text:
        return False
    pattern_len = max(1, int(min_pattern_chars))
    repeats = max(2, int(min_repeats))
    tail_len = pattern_len * repeats
    if len(response_text) < tail_len:
        return False
    tail = response_text[-tail_len:]
    pattern = tail[-pattern_len:]
    if not pattern.strip():
        return False
    return tail == pattern * repeats


def _evaluate_token_rate_guardrail(
    chunk_history: list[tuple[float, str]],
    max_token_rate: int,
    window_seconds: int,
    model_name: str | None = None,
) -> bool:
    """Evaluate whether token generation rate exceeds threshold over a rolling window.

    Computes tokens/second from the chunk history using ``count_text_tokens``
    and triggers only when the rate has been sustained over the *full* window
    duration (i.e. the window is fully populated with data).

    Args:
        chunk_history: List of ``(timestamp, chunk_text)`` tuples in chronological
            order. Timestamps are from ``time.monotonic()``.
        max_token_rate: Maximum allowed tokens per second. 0 or negative = disabled.
        window_seconds: Duration of the rolling window in seconds.
        model_name: Optional model name passed to ``count_text_tokens``.

    Returns:
        True if the sustained token rate exceeds ``max_token_rate`` over the
        full window; False otherwise.
    """
    if max_token_rate <= 0 or not chunk_history or len(chunk_history) < 2:
        return False

    from proxy.utils import count_text_tokens

    now = chunk_history[-1][0]
    window_start = now - window_seconds

    # Only consider chunks within the rolling window
    window_chunks = [(t, text) for t, text in chunk_history if t >= window_start]

    if len(window_chunks) < 2:
        return False

    # Compute total tokens and elapsed time over the window
    total_tokens = sum(
        count_text_tokens(text, model_name) for _, text in window_chunks
    )
    elapsed = window_chunks[-1][0] - window_chunks[0][0]

    if elapsed <= 0:
        return False

    tokens_per_second = total_tokens / elapsed

    # Only trigger once the window is fully populated (elapsed >= window_seconds)
    # to avoid false positives on short bursts.
    if elapsed >= window_seconds and tokens_per_second > max_token_rate:
        return True

    return False


def evaluate_stream_guardrail(
    runtime_seconds: float,
    completion_tokens: int,
    response_text: str,
    max_runtime_seconds: float | None,
    max_completion_tokens: int | None,
    repetition_min_pattern_chars: int,
    repetition_min_repeats: int,
    # Token-rate guardrail parameters (new in token-rate feature)
    chunk_history: list[tuple[float, str]] | None = None,
    max_token_rate: int = 0,
    token_rate_window_seconds: int = 5,
) -> str | None:
    """Evaluate whether the stream should be stopped due to guardrail violations.

    Priority order:
    1. Runtime cutoff - indicates a true runaway loop
    2. Repetition detection - indicates the model is stuck in a loop
    3. Token-rate guardrail - monitors tokens/second over a rolling window
       (only evaluated when enabled, i.e. ``max_token_rate > 0``)

    Note: The hard completion_tokens cutoff has been removed. Legitimate long
    responses should not be cut off. Loop detection via repetition check is
    used instead to catch runaway generation.
    """
    if max_runtime_seconds and runtime_seconds >= max_runtime_seconds:
        return "runtime"
    if _should_cutoff_for_repetition(response_text, repetition_min_pattern_chars, repetition_min_repeats):
        return "repetition"
    if max_token_rate > 0 and _evaluate_token_rate_guardrail(
        chunk_history or [],
        max_token_rate,
        token_rate_window_seconds,
    ):
        return "token_rate"
    return None


def _should_invalidate_on_guardrail(
    guardrail_reason: str | None,
    invalidate_on_cutoff: bool,
    invalidate_on_repetition: bool,
) -> bool:
    """Determine whether a session should be invalidated due to a guardrail.

    By default:
    - "runtime" guardrail invalidates the session (indicates true runaway loop)
    - "repetition" guardrail does NOT invalidate by default (let client retry)
    - "token_rate" guardrail does NOT invalidate (consistent with repetition)
    - "completion_tokens" guardrail never invalidates (removed in favor of loop detection)
    """
    if not guardrail_reason:
        return False
    if guardrail_reason == "repetition":
        return bool(invalidate_on_repetition)
    # token_rate guardrail does not invalidate by default (consistent with repetition)
    if guardrail_reason == "token_rate":
        return False
    # completion_tokens reason should never cause invalidation
    # (it's no longer a guardrail reason, but handle defensively)
    if guardrail_reason == "completion_tokens":
        return False
    return bool(invalidate_on_cutoff)


# ===================================================================
# Session history helpers
# ===================================================================


def merge_session_history_for_update(
    existing_messages: list[dict[str, Any]],
    request_messages: list[dict[str, Any]],
    delta_messages: list[dict[str, Any]] | None,
    is_delta_request: bool,
    assistant_content: str | None,
    assistant_message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if is_delta_request and delta_messages:
        merged = list(existing_messages) + list(delta_messages)
    else:
        merged = list(request_messages)

    if assistant_message is not None:
        merged.append(dict(assistant_message))
    elif assistant_content:
        if (
            not merged
            or merged[-1].get("role") != "assistant"
            or merged[-1].get("content") != assistant_content
        ):
            merged.append({"role": "assistant", "content": assistant_content})
    return merged


# ===================================================================
# Delta routing classification
# ===================================================================


def _classify_delta_routing(
    history_matches: bool,
    delta_message_count: int,
    restore_confirmed: bool,
    require_restore_signal: bool = True,
    force_full_prompt: bool = False,
) -> tuple[bool, str | None]:
    """Decide whether to use delta routing.

    When ``require_restore_signal`` is True, delta routing requires explicit
    restore confirmation from backend signals/logs.

    Returns (use_delta, reason) where reason is None when delta can be used,
    or a string explaining why a full re-process is required.
    """
    if not history_matches:
        logger.warning(
            "cache_invalidation: full re-process required - history_mismatch"
        )
        return False, "history_mismatch"
    if delta_message_count <= 0:
        logger.warning(
            "cache_invalidation: full re-process required - no_new_messages "
            "(delta_message_count=%s)", delta_message_count
        )
        return False, "no_new_messages"
    if force_full_prompt:
        logger.info(
            "cache_invalidation: full re-process required - "
            "delta_disabled (force_full_prompt=True)"
        )
        return False, "delta_disabled"
    if require_restore_signal and not restore_confirmed:
        logger.warning(
            "cache_invalidation: full re-process required - "
            "missing_restore_signal"
        )
        return False, "missing_restore_signal"
    return True, None


def _has_explicit_restore_signal(
    response_headers: dict[str, str],
    response_json: dict[str, Any] | None = None,
) -> bool:
    """Return True only when explicit backend restore evidence is present."""
    header_candidates = {
        "x-llama-session-restored",
        "x-session-restored",
        "x-llama-cache-restored",
        "x-kv-cache-restored",
        "x-cache-restored",
    }
    for key, value in response_headers.items():
        if key.lower() not in header_candidates:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "restored", "hit"}:
            return True

    if isinstance(response_json, dict):
        for field in (
            "session_restored",
            "cache_restored",
            "restore_success",
            "kv_cache_restored",
        ):
            if response_json.get(field) is True:
                return True
    return False


# ===================================================================
# Session header resolution
# ===================================================================


def _resolve_session_id_header(
    headers: dict,
) -> tuple[str | None, str | None]:
    """Resolve a session identifier from request headers.

    Priority order:
    1. ``X-Session-Id``
    2. ``session_id``
    3. ``X-Client-Request-Id``
    4. ``X-Session-Affinity``

    Returns ``(session_id, source_header_name)``.
    """
    for header_name in ("x-session-id", "session_id", "x-client-request-id", "x-session-affinity"):
        value = headers.get(header_name)
        if value:
            return value, header_name
    return None, None


def _log_session_header_resolution(
    session_id_header: str | None,
    header_source: str | None,
) -> None:
    """Log whether a session header was provided on the request."""
    try:
        srv = _srv()
        if header_source:
            prefix = session_id_header[:8] if session_id_header else "unknown"
            srv.logger.info(
                "Session header resolved: source=%s session=%s...",
                header_source,
                prefix,
            )
        else:
            srv.logger.info("No session header provided; proxy will generate session id")
    except Exception:
        pass
