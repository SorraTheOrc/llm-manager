import io
import logging
import json
import pytest

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


@pytest.mark.parametrize("chunk,expected", [
    (b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', "hello"),
    (b'data: {"choices":[{"delta":{"content":"first\\n"}}]}\n\n', "first\n"),
])
def test_console_prints_delta_content(tmp_path, chunk, expected):
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    server.log_response_chunk(chunk)

    out = strio.getvalue()
    assert out == expected


def test_console_fallback_prints_original_chunk(tmp_path):
    logger, ch, strio = _configure_logger_for_test(tmp_path)

    chunk = b"this-is-not-json"
    server.log_response_chunk(chunk)

    out = strio.getvalue()
    # Fallback should write the original chunk (decoded)
    assert out == chunk.decode("utf-8")
