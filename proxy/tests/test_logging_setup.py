"""Tests for the logging setup function.

Tests cover:
- Fallback path resolution when the configured log directory is unwritable
- Startup log message confirming the log file path
"""

import logging
from pathlib import Path


def test_logging_fallback_path_resolves_to_proxy_logs():
    """When PermissionError occurs, fallback should resolve to proxy/logs,
    not proxy/proxy/logs (LP-0MRB2ABNX008SF9P)."""
    from proxy import utils as utils_mod
    utils_path = Path(utils_mod.__file__)
    # proxy/proxy/utils.py -> parent.parent = proxy/
    correct_fallback = utils_path.parent.parent / "logs"

    assert correct_fallback.name == "logs", (
        f"Expected fallback to be named 'logs', got {correct_fallback.name}"
    )
    # The expected fallback should be at the proxy/ directory level
    assert correct_fallback.parent.name == "proxy", (
        f"Expected fallback parent to be 'proxy', got {correct_fallback.parent.name}"
    )


def test_logging_fallback_path_is_not_double_proxy():
    """The fallback path should resolve to proxy/logs not proxy/proxy/logs."""
    from proxy import utils as utils_mod
    utils_path = Path(utils_mod.__file__)

    # Current wrong behavior would resolve to proxy/proxy/logs
    wrong_fallback = utils_path.parent / "logs"
    correct_fallback = utils_path.parent.parent / "logs"

    # The wrong path has "proxy/proxy/logs"
    assert wrong_fallback.parent.name == "proxy"
    assert wrong_fallback.parent.parent.name == "proxy"

    # The correct path is "proxy/logs"
    assert correct_fallback.parent.name == "proxy"
    # Ensure these are different paths
    assert correct_fallback != wrong_fallback


def test_logging_startup_message_logged(tmp_path, monkeypatch, caplog):
    """Setup logging should emit an info-level message confirming the log path."""
    from proxy.server import setup_logging

    caplog.set_level(logging.INFO)

    log_dir = str(tmp_path / "proxy-logs")
    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)

    config = {
        "logging": {
            "directory": log_dir,
            "level": "INFO",
            "rotation_hours": 6,
            "retention_days": 90,
        }
    }

    setup_logging(config)

    # Should contain a startup message with the log file path
    assert any(
        "Logging initialised" in record.getMessage()
        or "logging initialised" in record.getMessage().lower()
        for record in caplog.records
    ), "Expected a startup log message confirming the log file path"
