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
