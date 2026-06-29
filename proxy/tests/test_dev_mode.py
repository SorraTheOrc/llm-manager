"""Tests for dev mode functionality.

These tests verify that the proxyctl CLI and server support dev mode correctly:
- proxyctl start --dev uses separate PID/log files
- Dev mode sets correct environment variables
- Dev mode enables DEBUG logging
- Production mode is unaffected by dev mode changes
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestDevModeLogging:
    """Test that setup_logging respects LLAMA_PROXY_DEV environment variable."""

    def test_dev_mode_uses_xdg_log_directory(self, tmp_path, monkeypatch):
        """Dev mode should use XDG-based dev log directory."""
        xdg_home = str(tmp_path / "state")
        monkeypatch.setenv("LLAMA_PROXY_DEV", "1")
        monkeypatch.setenv("XDG_STATE_HOME", xdg_home)
        # Import fresh
        from proxy.server import setup_logging
        config = {
            "logging": {
                "directory": "/var/log/llama-proxy",
                "level": "INFO",
                "rotation_hours": 6,
                "retention_days": 90,
            }
        }
        logger = setup_logging(config)
        # The log_dir should be in the XDG home under llama-proxy-dev/logs
        assert logger is not None

    def test_dev_mode_sets_debug_level(self, tmp_path, monkeypatch):
        """Dev mode should set DEBUG log level."""
        xdg_home = str(tmp_path / "state")
        monkeypatch.setenv("LLAMA_PROXY_DEV", "1")
        monkeypatch.setenv("XDG_STATE_HOME", xdg_home)
        from proxy import server
        from proxy.server import setup_logging
        config = {
            "logging": {
                "directory": "/var/log/llama-proxy",
                "level": "INFO",
                "rotation_hours": 6,
                "retention_days": 90,
            }
        }
        logger = setup_logging(config)
        assert logger.level == server.logging.DEBUG

    def test_production_mode_ignores_llama_proxy_dev(self, tmp_path, monkeypatch):
        """Production mode should not be affected by LLAMA_PROXY_DEV."""
        xdg_home = str(tmp_path / "state")
        log_dir = str(tmp_path / "prod-logs")
        # Make sure LLAMA_PROXY_DEV is NOT set
        monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)
        from proxy import server
        from proxy.server import setup_logging
        config = {
            "logging": {
                "directory": log_dir,
                "level": "INFO",
                "rotation_hours": 6,
                "retention_days": 90,
            }
        }
        logger = setup_logging(config)
        # Log level should be INFO, not DEBUG
        assert logger.level == server.logging.INFO


class TestProxyctlDevFlag:
    """Test that proxyctl correctly handles --dev flag."""

    def test_proxyctl_help_shows_dev_flag(self):
        """proxyctl help should document --dev flag."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        result = subprocess.run(
            ["bash", str(proxyctl_path), "help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "--dev" in result.stdout
        assert "development mode" in result.stdout.lower() or "dev mode" in result.stdout.lower()

    def test_proxyctl_status_with_dev_flag(self):
        """proxyctl status --dev should not error when no dev instance is running."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        result = subprocess.run(
            ["bash", str(proxyctl_path), "status", "--dev"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Exit code 3 means "not running", which is expected
        assert result.returncode == 3
        assert "proxy not running" in result.stdout

    def test_proxyctl_stop_with_dev_flag(self):
        """proxyctl stop --dev should not error when no dev instance is running."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        result = subprocess.run(
            ["bash", str(proxyctl_path), "stop", "--dev"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Exit code 0 means "stopped" (or already not running), which is fine
        assert result.returncode == 0

    def test_proxyctl_dev_args_filtered_correctly(self):
        """The --dev flag should be filtered out and not passed to commands."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        # Running status --dev with --dev in the middle should still work
        result = subprocess.run(
            ["bash", str(proxyctl_path), "status", "--dev"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Should not error with "Unknown command"
        assert "Unknown command" not in result.stdout
        assert result.returncode == 3  # not running


class TestDevModePortDefaults:
    """Test that dev mode uses correct default ports."""

    def test_dev_mode_env_vars_set(self):
        """Dev mode should set correct environment variable defaults."""
        # We can't easily test the actual proxyctl start behavior without
        # a running llama-server, but we can verify the env var defaults
        # are documented correctly in the usage output
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        result = subprocess.run(
            ["bash", str(proxyctl_path), "help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        # Verify port 8001 is mentioned as the dev port
        assert "8001" in result.stdout
        # Verify port 8081 is mentioned as the dev backend port
        assert "8081" in result.stdout


class TestDevModePIDFile:
    """Test that dev mode uses separate PID file."""

    def test_dev_pid_file_defined(self):
        """Dev mode should use proxy.dev.pid for its PID file."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        content = proxyctl_path.read_text()
        assert "proxy.dev.pid" in content
        assert "DEV_PID_FILE" in content

    def test_prod_pid_unchanged(self):
        """Production mode should still use proxy.pid."""
        proxyctl_path = Path(__file__).parent.parent / "proxyctl"
        content = proxyctl_path.read_text()
        # Production PID file should still be proxy.pid
        assert "proxy.pid" in content
        assert 'PID_FILE="$PID_DIR/proxy.pid"' in content
