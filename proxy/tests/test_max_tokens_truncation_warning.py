"""
Tests for max_tokens truncation warning (LP-0MS4C6E2L004HLLZ).

Verifies that:
1. A WARNING-level log is emitted when ``finish_reason: "length"`` is received
   from the upstream provider, indicating truncation.
2. The config default for ``session_guardrail_max_completion_tokens`` defaults to
   16384 (was 2048).
3. The ``max_completion_tokens`` value in ``_get_guardrail_config`` defaults to
   16384.
"""

import io
import logging
import re
from pathlib import Path

import pytest
import proxy.server as server
from proxy.router import _get_guardrail_config


def _configure_logger_for_test(tmp_path: Path):
    """Set up a logger with a StringIO handler for assertion."""
    cfg = {
        "logging": {
            "directory": str(tmp_path / "logs"),
            "rotation_hours": 1,
            "retention_days": 1,
            "level": "INFO",
        }
    }

    logger = logging.getLogger("llama-proxy")
    for h in list(logger.handlers):
        logger.removeHandler(h)

    server.setup_logging(cfg)

    # Capture all log output via a StringIO handler
    strio = io.StringIO()
    handler = logging.StreamHandler(strio)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)

    # Also capture the ContentOnlyConsoleHandler output
    console_handler = None
    for h in logger.handlers:
        if isinstance(h, server.ContentOnlyConsoleHandler):
            console_handler = h
            break
    # Set it to also use our strio for capturing
    if console_handler:
        console_handler.setStream(strio)

    strio.truncate(0)
    strio.seek(0)
    return logger, strio


# ── Tests for finish_reason: "length" warning ─────────────────────────────


def test_finish_reason_length_logs_warning(tmp_path):
    """A WARNING-level log is emitted when finish_reason is 'length'."""
    logger, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"length"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    assert "WARNING" in out or "WARN" in out.upper()
    assert "truncat" in out.lower() or "length" in out.lower()


def test_finish_reason_stop_does_not_log_warning(tmp_path):
    """No warning log is emitted when finish_reason is 'stop' (normal)."""
    logger, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    out = strio.getvalue()
    # The Stream finished line is INFO level, not WARNING — should not be captured
    # by our WARNING-level handler
    assert "truncat" not in out.lower()


def test_finish_reason_length_includes_token_info(tmp_path):
    """Truncation warning includes token usage info when available."""
    logger, strio = _configure_logger_for_test(tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"length"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":2048,"total_tokens":2058}}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess456", model="test-model")

    out = strio.getvalue()
    assert "WARNING" in out
    assert "2048" in out


# ── Tests for config default change ───────────────────────────────────────


def test_guardrail_config_defaults_to_16384():
    """_get_guardrail_config defaults max_completion_tokens to 16384."""
    config = _get_guardrail_config({})
    assert config["max_completion_tokens"] == 16384


def test_guardrail_config_override_still_works():
    """_get_guardrail_config respects explicit config overrides."""
    config = _get_guardrail_config({"session_guardrail_max_completion_tokens": 8192})
    assert config["max_completion_tokens"] == 8192


def test_guardrail_config_zero_uses_default():
    """_get_guardrail_config falls back to default when value is 0."""
    config = _get_guardrail_config({"session_guardrail_max_completion_tokens": 0})
    assert config["max_completion_tokens"] == 16384
