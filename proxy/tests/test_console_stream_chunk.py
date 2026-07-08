import io
import logging
import json
import pytest
import re

from pathlib import Path

import proxy.server as server


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


@pytest.mark.parametrize("chunk,expected", [
    (b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', "hello"),
    (b'data: {"choices":[{"delta":{"content":"first\\n"}}]}\n\n', "first\n"),
])
def test_console_prints_delta_content(tmp_path, chunk, expected):
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Content is wrapped in bold ANSI codes, so strip them for comparison
    assert _strip_ansi(out) == expected


def test_console_suppresses_non_json_chunks(tmp_path):
    """Non-JSON chunks should not display raw content in console."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b"this-is-not-json"
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Raw content should NOT appear in console - only extracted delta.content
    assert out == ""


def test_console_formats_reasoning_content_as_dim(tmp_path):
    """reasoning_content should be displayed with dim/grey formatting."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Should contain DIM ANSI code (\x1b[2m) for reasoning_content
    assert '\x1b[2m' in out
    assert 'thinking' in _strip_ansi(out)


def test_console_formats_content_as_bold(tmp_path):
    """content should be displayed with bold formatting."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Should contain BOLD ANSI code (\x1b[1m) for content
    assert '\x1b[1m' in out
    assert 'hello' in _strip_ansi(out)


# --- Tests for finish_reason / trailing newline and stop-reason logging ---


def test_console_appends_newline_on_finish_reason(tmp_path):
    """When the final chunk has finish_reason, append newline after content."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk)

    out = _strip_ansi(strio.getvalue())
    # The content "hello" should be followed by a newline before the log line
    assert "hello\n" in out, f"Expected newline after content, got: {out!r}"


def test_console_no_newline_without_finish_reason(tmp_path):
    """No trailing newline when finish_reason is absent."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = _strip_ansi(strio.getvalue())
    # Content should appear as-is without an appended newline
    assert out == "hello"


def test_console_newline_for_finish_reason_only_chunk(tmp_path):
    """Even a finish_reason-only chunk (no delta.content) produces a newline."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Should contain at least a newline somewhere in the output
    assert "\n" in out, f"Expected a newline in output, got: {out!r}"


def test_stop_reason_logged_on_finish_reason(tmp_path):
    """Logger emits a stop-reason line when finish_reason is present."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert "Stream finished: reason=stop" in out


def test_stop_reason_with_usage(tmp_path):
    """Stop-reason line includes token usage when usage object is present."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert "Stream finished: reason=stop tokens=10/20/30" in out


def test_stop_reason_no_usage(tmp_path):
    """Stop-reason line omits tokens when usage is absent."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert "Stream finished: reason=stop" in out
    assert "tokens=" not in out


def test_no_stop_reason_without_finish_reason(tmp_path):
    """No stop-reason log line when finish_reason is absent."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert "Stream finished:" not in out


def test_done_sentinel_no_side_effects(tmp_path):
    """[DONE] sentinel events produce no spurious newlines or log lines."""
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b'data: [DONE]\n\n'
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # No content to display, no finish_reason to log
    assert out == "" or "Stream finished:" not in out
