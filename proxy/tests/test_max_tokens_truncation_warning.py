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

import logging
import re
from pathlib import Path

import proxy.server as server
import pytest
from proxy.router import _get_guardrail_config

# ── Tests for finish_reason: "length" warning ─────────────────────────────


def _caplog_setup(caplog, tmp_path: Path, log_level=logging.WARNING) -> None:
    """Configure logging and attach caplog to capture the llama-proxy logger.

    Sets up logging via server.setup_logging then configures caplog
    to intercept the ``llama-proxy`` logger at the desired level.
    """
    cfg = {
        "logging": {
            "directory": str(tmp_path / "logs"),
            "rotation_hours": 1,
            "retention_days": 1,
            "level": "INFO",
        }
    }
    server.setup_logging(cfg)
    caplog.set_level(log_level, logger="llama-proxy")


def test_finish_reason_length_logs_warning(caplog, tmp_path):
    """A WARNING-level log is emitted when finish_reason is 'length'."""
    _caplog_setup(caplog, tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"length"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    assert len(caplog.records) > 0, "Expected at least one log record"
    # At least one record should be at WARNING level with truncation-related text
    assert any(
        r.levelname in ("WARNING", "WARN") and (
            "truncat" in r.getMessage().lower() or "length" in r.getMessage().lower()
        )
        for r in caplog.records
    ), "Expected a WARNING-level log about response truncation"


def test_finish_reason_stop_does_not_log_warning(caplog, tmp_path):
    """No warning log is emitted when finish_reason is 'stop' (normal)."""
    _caplog_setup(caplog, tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    # The Stream finished line is INFO level — should not appear as WARNING
    assert not any(
        r.levelname in ("WARNING", "WARN")
        for r in caplog.records
    ), "Expected no WARNING-level log for finish_reason=stop"


def test_finish_reason_length_includes_token_info(caplog, tmp_path):
    """Truncation warning includes token usage info when available."""
    _caplog_setup(caplog, tmp_path)

    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"length"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":2048,"total_tokens":2058}}\n\n'
    )
    server.log_response_chunk(chunk, session_id="sess456", model="test-model")

    assert any(
        r.levelname in ("WARNING", "WARN") and "2048" in r.getMessage()
        for r in caplog.records
    ), "Expected a WARNING log containing token count 2048"


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
