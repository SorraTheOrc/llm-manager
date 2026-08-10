"""Tests for session-list index observability (LP-0MSNM9IAC000GVXT).

Verifies:
1. SessionRecorder tracks index-hit vs cold-scan-fallback counts and the
   last cold-scan duration.
2. /admin/metrics exposes the index observability block.
3. Observability does not change behaviour (counters are additive).
"""

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_recording_dir(tmp_path):
    """Provide a temporary directory for recording files."""
    d = tmp_path / "session-recordings"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def recorder(temp_recording_dir):
    """Return a SessionRecorder instance configured with a temp directory."""
    from proxy.session_recorder import SessionRecorder
    return SessionRecorder(recording_path=temp_recording_dir)


class TestIndexObservability:
    """Index size / hit / scan counters on SessionRecorder."""

    @pytest.mark.asyncio
    async def test_warm_list_sessions_increments_index_hit(self, recorder, temp_recording_dir):
        """A warm list_sessions call counts as an index hit, not a scan."""
        await recorder.record_request(
            "sess-obs-1", "client_to_proxy",
            {"messages": [{"role": "user", "content": "hello"}]},
            model="qwen3", provider="local",
        )

        before = recorder.get_index_observability()
        recorder.list_sessions()
        after = recorder.get_index_observability()

        assert after["index_hits"] == before["index_hits"] + 1
        assert after["cold_scans"] == before["cold_scans"]

    @pytest.mark.asyncio
    async def test_cold_list_sessions_increments_scan(self, recorder, temp_recording_dir):
        """A cold list_sessions (empty index) records a cold scan."""
        before = recorder.get_index_observability()
        # No recordings yet → index empty → cold scan runs once.
        recorder.list_sessions()
        after = recorder.get_index_observability()

        assert after["cold_scans"] == before["cold_scans"] + 1

    @pytest.mark.asyncio
    async def test_scan_duration_recorded(self, recorder, temp_recording_dir):
        """Cold-scan duration is captured in the observability block."""
        recorder.list_sessions()
        after = recorder.get_index_observability()

        # Duration is a float >= 0; recorded when a scan actually runs.
        assert "last_scan_duration_seconds" in after
        assert isinstance(after["last_scan_duration_seconds"], float)

    def test_index_size_reported(self, recorder, temp_recording_dir):
        """Index size is reported (>= 0)."""
        obs = recorder.get_index_observability()
        assert "index_size" in obs
        assert obs["index_size"] >= 0


class TestAdminMetricsIndexBlock:
    """admin_metrics exposes the index observability block."""

    @pytest.mark.asyncio
    async def test_admin_metrics_includes_index_observability(self, monkeypatch):
        from proxy.handlers import admin_metrics

        mock_srv = MagicMock()
        mock_srv.config = {"server": {}}
        mock_srv.metrics = MagicMock()
        mock_srv.session_manager = MagicMock()
        mock_srv.session_manager.get_metrics.return_value = {}
        mock_srv.session_restore_observability = {}
        mock_srv.session_single_flight_observability = {}
        mock_srv.session_guardrail_observability = {}
        mock_srv.model_last_used = {}

        recorder = MagicMock()
        recorder.get_index_observability.return_value = {
            "index_size": 3,
            "index_hits": 5,
            "cold_scans": 1,
            "last_scan_duration_seconds": 0.01,
        }
        mock_srv.session_recorder = recorder

        with patch("proxy.handlers._srv", return_value=mock_srv):
            result = await admin_metrics()

        assert "index_observability" in result
        assert result["index_observability"]["index_size"] == 3
        assert result["index_observability"]["index_hits"] == 5
        assert result["index_observability"]["cold_scans"] == 1
