import io
import logging
import re
from pathlib import Path

import proxy.server as server
from proxy.router_helpers import _get_request_preview


def _configure_logger_for_test(tmp_path: Path):
    # Prepare a minimal logging config that writes to tmp_path
    cfg = {"logging": {"directory": str(tmp_path / "logs"), "rotation_hours": 1, "retention_days": 1, "level": "INFO"}}

    logger = logging.getLogger("llama-proxy")
    # Remove existing handlers to keep tests hermetic
    for h in list(logger.handlers):
        logger.removeHandler(h)

    server.setup_logging(cfg)

    # Find our ContentOnlyConsoleHandler
    console_handler = None
    for h in logger.handlers:
        if isinstance(h, server.ContentOnlyConsoleHandler):
            console_handler = h
            break
    assert console_handler is not None, "ContentOnlyConsoleHandler not installed"

    strio = io.StringIO()
    console_handler.setStream(strio)
    # clear anything that might have been written during setup
    strio.truncate(0)
    strio.seek(0)
    return logger, console_handler, strio


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# --- Tests for content suppression in console (LP-0MR90HJED005WI1Z) ---


def test_console_suppresses_streaming_content(tmp_path):
    """Streaming content (delta.content) should NOT appear in console output."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # No content should appear in console
    assert out == ""


def test_console_suppresses_reasoning_content(tmp_path):
    """reasoning_content should NOT appear in console output."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # No reasoning content should appear in console
    assert out == ""


def test_console_suppresses_non_json_chunks(tmp_path):
    """Non-JSON chunks should not display raw content in console."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b"this-is-not-json"
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert out == ""


def test_console_passes_through_lifecycle_lines(tmp_path):
    """Lifecycle log lines (Stream started/finished/error) still appear in console."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    # Log a non-STREAM_CHUNK line that the handler should pass through
    logger.info("Stream started: provider=test model=test-model session=abc123 request=hello")
    logger.info("Stream finished: reason=stop tokens=10/20/30 session=abc123 provider=test model=test-model request=hello")

    out = strio.getvalue()
    assert "Stream started:" in out
    assert "Stream finished:" in out


# --- Tests for Stream finished enhanced log lines ---


def test_stop_reason_logged_on_finish_reason(tmp_path):
    """Logger emits a stop-reason line when finish_reason is present."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    assert "Stream finished: reason=stop" in out


def test_stop_reason_with_usage(tmp_path):
    """Stop-reason line includes token usage when usage object is present."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    assert "Stream finished: reason=stop tokens=10/20/30" in out


def test_stop_reason_no_usage(tmp_path):
    """Stop-reason line omits tokens when usage is absent."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    assert "Stream finished: reason=stop" in out
    assert "tokens=" not in out


def test_no_stop_reason_without_finish_reason(tmp_path):
    """No stop-reason log line when finish_reason is absent."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    assert "Stream finished:" not in out


def test_done_sentinel_no_side_effects(tmp_path):
    """[DONE] sentinel events produce no spurious log lines."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: [DONE]\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert out == "" or "Stream finished:" not in out


# --- Tests for enhanced Stream finished with session/model info ---


def test_stop_reason_includes_session_and_model(tmp_path):
    """Stop-reason line includes session and model when provided."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk, session_id="sess456", model="qwen3", provider="local")

    out = strio.getvalue()
    assert "session=sess456" in out
    assert "provider=local" in out
    assert "model=qwen3" in out


def test_stop_reason_with_request_preview(tmp_path):
    """Stop-reason line includes request preview when body_json is provided."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    body_json = {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}]}
    server.log_response_chunk(chunk, session_id="sess456", model="qwen3", provider="local", body_json=body_json)

    out = strio.getvalue()
    assert "request=What is the capital of France?" in out


def test_stop_reason_with_long_request_preview_truncation(tmp_path):
    """Request preview is truncated to 80 chars with ... when longer."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    long_msg = "X" * 100
    body_json = {"messages": [{"role": "user", "content": long_msg}]}
    server.log_response_chunk(chunk, session_id="sess456", model="qwen3", provider="local", body_json=body_json)

    out = strio.getvalue()
    assert ("X" * 80) + "..." in out


# --- Tests for _get_request_preview helper ---


def test_get_request_preview_first_non_system():
    """Preview extracts first 80 chars of first non-system message."""
    body_json = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    }
    preview = _get_request_preview(body_json)
    assert preview == "What is the capital of France?"


def test_get_request_preview_truncates_long():
    """Preview truncates long messages to 80 chars with ..."""
    body_json = {
        "messages": [
            {"role": "user", "content": "A" * 100},
        ]
    }
    preview = _get_request_preview(body_json)
    assert preview == ("A" * 80) + "..."
    assert len(preview) == 83


def test_get_request_preview_short_message():
    """Preview returns full message if <= 80 chars."""
    body_json = {
        "messages": [
            {"role": "user", "content": "Hello"},
        ]
    }
    preview = _get_request_preview(body_json)
    assert preview == "Hello"


def test_get_request_preview_empty():
    """Preview returns empty string for empty/invalid input."""
    assert _get_request_preview({}) == ""
    assert _get_request_preview({"messages": []}) == ""
    assert _get_request_preview(None) == ""
    assert _get_request_preview({"messages": [{"role": "system", "content": "sys"}]}) == ""


def test_get_request_preview_skips_system():
    """Preview skips system messages and uses first non-system message."""
    body_json = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Always be concise."},
            {"role": "user", "content": "Hi there!"},
        ]
    }
    preview = _get_request_preview(body_json)
    assert preview == "Hi there!"


# ═══════════════════════════════════════════════════════════════════════════════
# LP-0MT6322OT00900OX: enriched stream-error payload in Stream finished lines
# ═══════════════════════════════════════════════════════════════════════════════


def test_error_finish_reason_includes_error_payload(tmp_path):
    """Stream finished: reason=error includes error type/message/action.

    When a synthetic finish_reason: error event carries an enriched error
    object (LP-0MSETOTWY000SU0Z), the log line must surface the payload
    so the proxy-usage analysis tool can classify it (LP-0MT6322OT00900OX).
    """
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices": [{"delta": {}, "finish_reason": "error", "error": '
        b'{"type": "stall_exhausted", "message": "Retry chain exhausted", '
        b'"provider": "opencode-go", "model": "deepseek-v4-flash", '
        b'"entry": "opencode-go-2-deepseek", "suggested_action": '
        b'"Check provider availability and retry"}}]}\n\n'
    )
    server.log_response_chunk(
        chunk, session_id="sess789", model="deepseek-v4-flash",
        provider="opencode-go", entry="opencode-go-2-deepseek",
    )

    out = strio.getvalue()
    assert "Stream finished: reason=error" in out
    assert "error_type=stall_exhausted" in out
    assert "error_message=Retry chain exhausted" in out
    assert "suggested_action=Check provider availability and retry" in out


def test_error_finish_reason_missing_payload_fields(tmp_path):
    """Error finish_reason without optional payload fields omits them."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    # Minimal enriched error (only type is present)
    chunk = (
        b'data: {"choices": [{"delta": {}, "finish_reason": "error", "error": '
        b'{"type": "stream_exception"}}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="s1")

    out = strio.getvalue()
    assert "Stream finished: reason=error" in out
    assert "error_type=stream_exception" in out
    assert "error_message=" not in out
    assert "suggested_action=" not in out


def test_error_finish_reason_without_error_object_omits_payload(tmp_path):
    """A finish_reason=error without an error object omits payload fields."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices": [{"delta": {}, "finish_reason": "error"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="s2")

    out = strio.getvalue()
    assert "Stream finished: reason=error" in out
    assert "error_type=" not in out
    assert "error_message=" not in out
    assert "suggested_action=" not in out


def test_non_error_finish_reason_no_error_payload(tmp_path):
    """Non-error finish_reasons must not carry error payload fields."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="s3")

    out = strio.getvalue()
    assert "Stream finished: reason=stop" in out
    assert "error_type=" not in out
    assert "error_message=" not in out


def test_error_finish_reason_includes_provider_model_entry(tmp_path):
    """Error log line also carries session/provider/model/entry params."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices": [{"delta": {}, "finish_reason": "error", "error": '
        b'{"type": "empty_response"}}]}\n\n'
    )
    server.log_response_chunk(
        chunk, session_id="sess01", model="deepseek-v4-flash",
        provider="opencode-go", entry="opencode-go-2-deepseek",
    )

    out = strio.getvalue()
    assert "error_type=empty_response" in out
    assert "session=sess01" in out
    assert "provider=opencode-go" in out
    assert "model=deepseek-v4-flash" in out
    assert "entry=opencode-go-2-deepseek" in out
