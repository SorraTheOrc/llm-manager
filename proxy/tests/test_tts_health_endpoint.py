"""
Tests for TTS status fields in the /health endpoint.

Verifies that the /health endpoint correctly reports:
- tts_server_running: bool (process alive via poll())
- tts_server_healthy: bool (HTTP probe to /health endpoint)
- tts_enabled: bool (from config)

These tests use mock objects to control server state without requiring
a real TTS server to be running.
"""

import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Note: _srv() is defined in proxy.handlers and proxy.backend_health
# Tests patch it via the handlers module directly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(tts_enabled=True, tts_server_port=8081, tts_server_host="localhost"):
    """Build a config dict with TTS settings."""
    return {
        "server": {
            "tts_enabled": tts_enabled,
            "tts_server_port": tts_server_port,
            "tts_server_host": tts_server_host,
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "llama_watchdog_interval_seconds": 5,
        },
        "models": {},
        "default_model": "test-model",
    }


def _make_mock_server(
    tts_process_alive=True,
    tts_health_ok=True,
    tts_enabled=True,
    tts_server_port=8081,
    tts_server_host="localhost",
    llama_process_alive=True,
    backend_ready=True,
    backend_reachable=True,
    self_healing=False,
    current_model="test-model",
):
    """Build a mock server object with the specified TTS/llama state."""
    srv = MagicMock()

    # Config
    config = _make_mock_config(tts_enabled, tts_server_port, tts_server_host)
    srv.config = config

    # TTS process
    if tts_process_alive:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)  # None means still running
        srv.tts_process = mock_proc
    else:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=1)  # non-None means exited
        srv.tts_process = mock_proc

    # TTS health probe result
    srv._tts_health_ok = tts_health_ok

    # Llama process
    if llama_process_alive:
        mock_llama = MagicMock(spec=subprocess.Popen)
        mock_llama.poll = MagicMock(return_value=None)
        srv.llama_process = mock_llama
    else:
        srv.llama_process = None

    # Backend state
    srv.backend_ready = backend_ready
    srv._probe_backend_reachable = AsyncMock(return_value=backend_reachable)
    srv._is_self_healing_active = MagicMock(return_value=self_healing)
    srv._backend_recovery_snapshot = MagicMock(return_value={})
    srv.backend_signal_counts = {}

    # Model state
    srv.current_model = current_model

    # Router models
    srv._extract_router_model_ids = MagicMock(return_value=None)

    return srv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpointTTSFields:
    """Test TTS status fields in /health endpoint."""

    def _call_health_check(self, srv_mock, probe_result=None):
        """Helper to call health_check with the given server mock.

        Args:
            srv_mock: The mock server object.
            probe_result: If set, overrides _probe_tts_health return value.
                          If None, uses srv_mock._tts_health_ok (default True).
        """
        from proxy.handlers import health_check

        expected_probe = probe_result if probe_result is not None else getattr(srv_mock, '_tts_health_ok', True)

        with patch("proxy.handlers._srv", return_value=srv_mock):
            with patch("proxy.handlers._probe_tts_health", return_value=expected_probe):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(health_check())
                finally:
                    loop.close()
        return result

    def test_health_includes_tts_fields_when_enabled_and_alive(self):
        """When TTS is enabled and process is alive and healthy, fields are True."""
        srv = _make_mock_server(
            tts_process_alive=True,
            tts_health_ok=True,
            tts_enabled=True,
        )

        result = self._call_health_check(srv)

        assert "tts_enabled" in result
        assert result["tts_enabled"] is True
        assert "tts_server_running" in result
        assert result["tts_server_running"] is True
        assert "tts_server_healthy" in result
        assert result["tts_server_healthy"] is True

    def test_health_includes_tts_fields_when_enabled_but_dead(self):
        """When TTS is enabled but process has exited, running=False."""
        srv = _make_mock_server(
            tts_process_alive=False,
            tts_health_ok=False,
            tts_enabled=True,
        )

        result = self._call_health_check(srv)

        assert result["tts_enabled"] is True
        assert result["tts_server_running"] is False
        assert result["tts_server_healthy"] is False

    def test_health_includes_tts_fields_when_enabled_but_unhealthy(self):
        """When TTS is enabled, process alive, but health probe fails."""
        srv = _make_mock_server(
            tts_process_alive=True,
            tts_health_ok=False,  # Process alive but probe fails
            tts_enabled=True,
        )

        result = self._call_health_check(srv)

        assert result["tts_enabled"] is True
        assert result["tts_server_running"] is True
        assert result["tts_server_healthy"] is False

    def test_health_includes_tts_fields_when_disabled(self):
        """When TTS is disabled, both running and healthy are False."""
        srv = _make_mock_server(
            tts_process_alive=False,
            tts_health_ok=False,
            tts_enabled=False,
        )

        result = self._call_health_check(srv)

        assert result["tts_enabled"] is False
        assert result["tts_server_running"] is False
        assert result["tts_server_healthy"] is False

    def test_health_endpoint_returns_all_fields(self):
        """The /health endpoint returns a complete set of fields."""
        srv = _make_mock_server()

        result = self._call_health_check(srv)

        expected_fields = [
            "status",
            "ready",
            "current_model",
            "loaded_models",
            "llama_server_running",
            "backend_reachable",
            "self_healing_in_progress",
            "backend_recovery",
            "backend_signals",
            "tts_enabled",
            "tts_server_running",
            "tts_server_healthy",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_health_status_is_healthy_when_all_ok(self):
        """When everything is healthy including TTS, status is 'healthy'."""
        srv = _make_mock_server(
            tts_process_alive=True,
            tts_health_ok=True,
            tts_enabled=True,
        )

        result = self._call_health_check(srv)

        assert result["status"] == "healthy"
        assert result["ready"] is True

    def test_health_status_not_affected_by_tts_state(self):
        """Overall /health status reflects proxy readiness, not TTS state.

        TTS being down does NOT change the overall proxy status — the TTS
        fields provide separate visibility into TTS subsystem health.
        """
        srv = _make_mock_server(
            tts_process_alive=False,
            tts_health_ok=False,
            tts_enabled=True,
        )

        result = self._call_health_check(srv, probe_result=False)

        # Status stays healthy because llama-server is fine
        assert result["status"] == "healthy"
        assert result["ready"] is True
        assert result["tts_server_running"] is False
        assert result["tts_server_healthy"] is False
