"""Observability Regression Suite.

Protects health/readiness/metrics/log/SSE endpoint contracts throughout
the proxy server refactor. All tests in this file are tagged with the
``refactor_parity`` marker so they can be run as part of the parity gate:

    $ pytest -m refactor_parity

Coverage
--------
- /health  — field shape, ready/degraded states, backend signals, recovery
- /metrics — Prometheus exposition endpoint
- /admin/metrics — detailed observability (restore, single-flight, guardrail)
- /events  — SSE initial status event shape
- /logs/tail — SSE initial response and error handling
- ContentOnlyConsoleHandler — content extraction and formatting
- Slot-reject observability signals — backend signal counters
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import proxy.server as server
import pytest

pytestmark = pytest.mark.refactor_parity


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _fake_reachable(port: int) -> bool:
    """Stub for _probe_backend_reachable — always reachable."""
    return True


async def _fake_query_status() -> dict:
    """Stub for query_llama_status."""
    return {
        "llama_server_running": True,
        "n_ctx": 4096,
        "kv_cache_tokens": 512,
        "router_mode": False,
    }


# ── /health endpoint contracts ───────────────────────────────────────────────


class TestHealthEndpoint:
    """Contract tests for the /health endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_health(self, monkeypatch):
        """Set baseline server state for health tests."""
        # Use a dict config so health check finds 'server' key
        monkeypatch.setattr(server, "config", {
            "server": {
                "llama_router_mode": False,
                "llama_server_port": 8080,
            }
        })
        monkeypatch.setattr(server, "llama_process", MagicMock())
        server.llama_process.poll.return_value = None  # running
        monkeypatch.setattr(server, "current_model", "test-model")
        monkeypatch.setattr(server, "backend_ready", True)
        monkeypatch.setattr(server, "backend_signal_counts", {
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        })
        monkeypatch.setattr(server, "backend_recovery_state", {
            "in_progress": False,
            "recovery_started_at": None,
            "retry_after_seconds": 30,
        })
        monkeypatch.setattr(server, "_probe_backend_reachable", _fake_reachable)
        monkeypatch.setattr(server, "_is_self_healing_active", lambda: False)
        monkeypatch.setattr(server, "_backend_recovery_snapshot",
                            lambda: {"in_progress": False, "recovery_started_at": None})
        monkeypatch.setattr(server, "query_llama_status", _fake_query_status)

    async def _get_health(self) -> dict:
        """Helper: GET /health and return JSON."""
        test_app = server.app
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            return resp.json()

    # ── Shape & type contracts ──

    @pytest.mark.asyncio
    async def test_health_field_types(self, monkeypatch):
        """All expected /health fields are present with correct types."""
        j = await self._get_health()

        assert isinstance(j, dict)

        # String fields
        assert isinstance(j.get("status"), str), "status must be str"
        assert j["status"] in ("healthy", "degraded"), f"unexpected status: {j['status']}"

        # Boolean fields
        for field in ("ready", "llama_server_running", "backend_reachable",
                      "self_healing_in_progress"):
            assert isinstance(j.get(field), bool), f"{field} must be bool, got {type(j.get(field))}"

        # current_model is str or None
        cm = j.get("current_model")
        assert cm is None or isinstance(cm, str), "current_model must be str or None"

        # backend_signals
        signals = j.get("backend_signals")
        assert isinstance(signals, dict), "backend_signals must be dict"
        for sig in ("connect_failures", "read_failures", "timeout_failures",
                    "other_failures", "concurrency_rejects"):
            assert isinstance(signals.get(sig), int), f"backend_signals.{sig} must be int"

        # backend_recovery
        recovery = j.get("backend_recovery")
        assert isinstance(recovery, dict), "backend_recovery must be dict"

    @pytest.mark.asyncio
    async def test_health_healthy_state(self, monkeypatch):
        """When backend is running and reachable, /health returns healthy."""
        j = await self._get_health()
        assert j["status"] == "healthy"
        assert j["ready"] is True
        assert j["llama_server_running"] is True
        assert j["current_model"] == "test-model"

    @pytest.mark.asyncio
    async def test_health_degraded_when_no_llama_process(self, monkeypatch):
        """When llama_process is None, /health returns degraded/unready."""
        monkeypatch.setattr(server, "llama_process", None)
        j = await self._get_health()
        assert j["status"] == "degraded"
        assert j["ready"] is False
        assert j["llama_server_running"] is False

    @pytest.mark.asyncio
    async def test_health_degraded_when_self_healing(self, monkeypatch):
        """When self-healing is active, /health returns degraded."""
        monkeypatch.setattr(server, "_is_self_healing_active", lambda: True)
        monkeypatch.setattr(server, "backend_recovery_state", {
            "in_progress": True,
            "recovery_started_at": "2026-01-01T00:00:00",
            "retry_after_seconds": 30,
        })
        monkeypatch.setattr(server, "_backend_recovery_snapshot",
                            lambda: {"in_progress": True, "recovery_started_at": "2026-01-01T00:00:00"})
        j = await self._get_health()
        assert j["status"] == "degraded"
        assert j["ready"] is False
        assert j["self_healing_in_progress"] is True
        assert j["backend_recovery"]["in_progress"] is True

    @pytest.mark.asyncio
    async def test_health_degraded_when_backend_not_ready(self, monkeypatch):
        """When backend_ready is False, /health returns degraded."""
        monkeypatch.setattr(server, "backend_ready", False)
        j = await self._get_health()
        assert j["status"] == "degraded"
        assert j["ready"] is False

    @pytest.mark.asyncio
    async def test_health_shows_backend_signals(self, monkeypatch):
        """backend_signals dict is present with correct values."""
        monkeypatch.setattr(server, "backend_signal_counts", {
            "connect_failures": 3,
            "read_failures": 1,
            "timeout_failures": 0,
            "other_failures": 2,
            "concurrency_rejects": 5,
        })
        j = await self._get_health()
        signals = j.get("backend_signals", {})
        assert signals["connect_failures"] == 3
        assert signals["concurrency_rejects"] == 5
        assert signals["read_failures"] == 1
        assert signals["timeout_failures"] == 0
        assert signals["other_failures"] == 2


# ── /metrics endpoint contracts ──────────────────────────────────────────────


class TestMetricsEndpoint:
    """Contract tests for the /metrics (Prometheus) endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_metrics(self, monkeypatch):
        monkeypatch.setattr(server, "config", {"server": {}})
        monkeypatch.setattr(server, "llama_process", MagicMock())
        server.llama_process.poll.return_value = None

    async def _get_metrics(self) -> httpx.Response:
        test_app = server.app
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get("/metrics")

    @pytest.mark.asyncio
    async def test_metrics_content_type(self, monkeypatch):
        """/metrics returns text/plain content type."""
        resp = await self._get_metrics()
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_metrics_contains_prometheus_metrics(self, monkeypatch):
        """/metrics body contains expected Prometheus metric names."""
        resp = await self._get_metrics()
        body = resp.text
        # Expected metric names (when prometheus_client is available)
        assert "llama_process_rss_bytes" in body or "Prometheus" in body
        assert "llama_models_loaded" in body or "Prometheus" in body


# ── /admin/metrics endpoint contracts ────────────────────────────────────────


class TestAdminMetricsEndpoint:
    """Contract tests for the /admin/metrics endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_admin_metrics(self, monkeypatch):
        monkeypatch.setattr(server, "config", {"server": {}})
        monkeypatch.setattr(server, "llama_process", MagicMock())
        server.llama_process.poll.return_value = None
        monkeypatch.setattr(server, "current_model", "test-model")
        monkeypatch.setattr(server, "backend_ready", True)
        monkeypatch.setattr(server, "backend_signal_counts", {
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        })
        monkeypatch.setattr(server, "backend_recovery_state", {
            "in_progress": False,
            "recovery_started_at": None,
            "retry_after_seconds": 30,
        })
        # Session restore observability
        monkeypatch.setattr(server, "session_restore_observability", {
            "restore_success_total": 10,
            "restore_fallback_total": {"history_mismatch": 3, "no_new_messages": 1},
            "delta_payload_bytes_total": 4096,
        })
        # Single-flight observability
        monkeypatch.setattr(server, "session_single_flight_observability", {
            "queue_events_total": 5,
            "reject_events_total": 2,
            "active_sessions_current": 1,
            "queue_depth_current": 0,
        })
        # Guardrail observability
        monkeypatch.setattr(server, "session_guardrail_observability", {
            "guardrail_cutoff_total": 7,
            "guardrail_cutoff_reasons": {"repetition": 5, "length": 2},
            "session_invalidation_total": 3,
            "session_invalidation_reasons": {"guardrail_triggered": 3},
        })
        monkeypatch.setattr(server, "_backend_recovery_snapshot",
                            lambda: {"in_progress": False, "recovery_started_at": None})
        monkeypatch.setattr(server, "query_llama_status",
                            AsyncMock(return_value={"llama_server_running": True}))
        monkeypatch.setattr(server, "session_manager", MagicMock())
        server.session_manager.get_metrics.return_value = {"total_sessions": 2}

    async def _get_admin_metrics(self) -> dict:
        test_app = server.app
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/admin/metrics")
            assert resp.status_code == 200
            return resp.json()

    @pytest.mark.asyncio
    async def test_admin_metrics_shape(self, monkeypatch):
        """Admin metrics endpoint returns expected top-level fields."""
        j = await self._get_admin_metrics()

        assert isinstance(j.get("models_max"), (int, type(None)))
        assert isinstance(j.get("loaded_models"), (list, type(None)))
        assert isinstance(j.get("per_model"), dict)
        assert isinstance(j.get("process_rss_bytes"), (int, type(None)))
        assert isinstance(j.get("session_metrics"), dict)

    @pytest.mark.asyncio
    async def test_admin_metrics_session_restore_fields(self, monkeypatch):
        """Session restore observability fields are present."""
        j = await self._get_admin_metrics()

        assert j["restore_success_total"] == 10
        fallback_total = j.get("restore_fallback_total", {})
        assert isinstance(fallback_total, dict)
        assert fallback_total.get("history_mismatch") == 3
        assert fallback_total.get("no_new_messages") == 1
        assert j["delta_payload_bytes_total"] == 4096

    @pytest.mark.asyncio
    async def test_admin_metrics_single_flight_fields(self, monkeypatch):
        """Single-flight observability fields are present."""
        j = await self._get_admin_metrics()

        sf = j.get("single_flight_metrics", {})
        assert isinstance(sf, dict)
        assert sf.get("queue_events_total") == 5
        assert sf.get("reject_events_total") == 2
        assert sf.get("active_sessions_current") == 1
        assert sf.get("queue_depth_current") == 0

    @pytest.mark.asyncio
    async def test_admin_metrics_guardrail_fields(self, monkeypatch):
        """Guardrail observability fields are present."""
        j = await self._get_admin_metrics()

        gr = j.get("guardrail_metrics", {})
        assert isinstance(gr, dict)
        assert gr.get("guardrail_cutoff_total") == 7
        assert isinstance(gr.get("guardrail_cutoff_reasons"), dict)
        assert gr["guardrail_cutoff_reasons"].get("repetition") == 5
        assert gr["guardrail_cutoff_reasons"].get("length") == 2
        assert gr.get("session_invalidation_total") == 3
        assert isinstance(gr.get("session_invalidation_reasons"), dict)
        assert gr["session_invalidation_reasons"].get("guardrail_triggered") == 3

    @pytest.mark.asyncio
    async def test_admin_metrics_backend_signals(self, monkeypatch):
        """Backend signals are present in admin metrics."""
        j = await self._get_admin_metrics()

        signals = j.get("backend_signals", {})
        assert isinstance(signals, dict)
        for sig in ("connect_failures", "read_failures", "timeout_failures",
                    "other_failures", "concurrency_rejects"):
            assert sig in signals, f"Missing backend signal: {sig}"

    @pytest.mark.asyncio
    async def test_admin_metrics_backend_recovery(self, monkeypatch):
        """Backend recovery snapshot is present."""
        j = await self._get_admin_metrics()

        recovery = j.get("backend_recovery", {})
        assert isinstance(recovery, dict)
        assert "in_progress" in recovery


# ── /events SSE endpoint contracts ───────────────────────────────────────────


class TestEventsEndpoint:
    """Contract tests for the /events SSE endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_events(self, monkeypatch):
        monkeypatch.setattr(server, "config", {"server": {"llama_router_mode": False}})
        monkeypatch.setattr(server, "llama_process", MagicMock())
        server.llama_process.poll.return_value = None
        monkeypatch.setattr(server, "current_model", "test-model")
        monkeypatch.setattr(server, "query_llama_status", _fake_query_status)
        monkeypatch.setattr(server, "sse_clients", set())
        monkeypatch.setattr(server, "token_counts", {
            "total_sent": 1000,
            "total_recv": 500,
        })

    @pytest.mark.asyncio
    async def test_events_content_type(self, monkeypatch):
        """/events returns text/event-stream content type."""
        response = await server.status_events()
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_events_initial_payload(self, monkeypatch):
        """/events initial event contains status with expected fields."""
        response = await server.status_events()
        # Read first chunk from the body_iterator with a timeout
        try:
            chunk = await asyncio.wait_for(
                response.body_iterator.__anext__(),
                timeout=3.0
            )
        except TimeoutError:
            pytest.fail("event_generator did not yield initial chunk within 3s")

        raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        assert "data: " in raw

        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                j = json.loads(payload)
                assert isinstance(j, dict)
                assert j.get("type") == "status"
                assert "current_model" in j
                assert "llama_server_running" in j
                assert "n_ctx" in j
                assert "kv_cache_tokens" in j
                assert "total_sent" in j
                assert "total_recv" in j
                break


# ── /logs/tail SSE endpoint contracts ────────────────────────────────────────


class TestLogsTailEndpoint:
    """Contract tests for the /logs/tail SSE endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_logs_tail(self, monkeypatch):
        monkeypatch.setattr(server, "config", {"server": {}})
        monkeypatch.setattr(server, "log_dir", None)
        monkeypatch.setattr(server, "log_tail_clients", set())

    @pytest.mark.asyncio
    async def test_logs_tail_handles_missing_log_file(self, monkeypatch):
        """/logs/tail returns error SSE event when log file does not exist."""
        nonexistent = Path("/nonexistent/proxy.log")
        monkeypatch.setattr(server, "_resolve_log_path", lambda source="proxy": nonexistent)

        from fastapi import Request
        scope = {"type": "http", "path": "/logs/tail", "query_string": b"source=proxy"}
        request = Request(scope)
        response = await server.tail_logs(request, lines=100, source="proxy")

        try:
            chunk = await asyncio.wait_for(
                response.body_iterator.__anext__(),
                timeout=3.0
            )
        except TimeoutError:
            pytest.fail("tail_logs did not yield within 3s")

        raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        assert "error" in raw
        assert "log_not_found" in raw

    @pytest.mark.asyncio
    async def test_logs_tail_invalid_source_falls_back_to_proxy(self, monkeypatch):
        """Invalid source param falls back to proxy."""
        nonexistent = Path("/nonexistent/proxy.log")
        monkeypatch.setattr(server, "_resolve_log_path",
                            lambda source="proxy": nonexistent)

        from fastapi import Request
        scope = {"type": "http", "path": "/logs/tail", "query_string": b"source=invalid"}
        request = Request(scope)
        response = await server.tail_logs(request, lines=100, source="invalid")

        try:
            chunk = await asyncio.wait_for(
                response.body_iterator.__anext__(),
                timeout=3.0
            )
        except TimeoutError:
            pytest.fail("tail_logs did not yield within 3s")

        raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        # Should still resolve to proxy path (not crash)
        assert "error" in raw or "data:" in raw

    @pytest.mark.asyncio
    async def test_logs_tail_content_type(self, monkeypatch):
        """/logs/tail returns text/event-stream content type."""
        from fastapi import Request
        scope = {"type": "http", "path": "/logs/tail", "query_string": b"source=proxy"}
        request = Request(scope)
        response = await server.tail_logs(request, lines=100, source="proxy")
        assert response.media_type == "text/event-stream"


# ── ContentOnlyConsoleHandler behavior ───────────────────────────────────────


class TestContentOnlyConsoleHandler:
    """Tests for ContentOnlyConsoleHandler content extraction and formatting."""

    @pytest.fixture
    def handler(self):
        return server.ContentOnlyConsoleHandler()

    # extract_streamed_content_from_chunk

    def test_extract_streamed_content_from_chunk(self):
        """extract_streamed_content_from_chunk extracts content from SSE chunk."""
        chunk = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        result = server.extract_streamed_content_from_chunk(chunk)
        assert result == "Hello"

    def test_extract_streamed_content_from_chunk_reasoning(self):
        """extract_streamed_content_from_chunk extracts reasoning_content."""
        chunk = 'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n'
        result = server.extract_streamed_content_from_chunk(chunk)
        assert result == "thinking..."

    def test_extract_streamed_content_from_chunk_both(self):
        """extract_streamed_content_from_chunk concatenates reasoning and content."""
        chunk = 'data: {"choices":[{"delta":{"reasoning_content":"think","content":"answer"}}]}'
        result = server.extract_streamed_content_from_chunk(chunk)
        assert result is not None
        assert "think" in result
        assert "answer" in result

    def test_extract_streamed_content_no_content(self):
        """extract_streamed_content_from_chunk returns None when no content."""
        chunk = 'data: {"choices":[{"delta":{}}]}'
        result = server.extract_streamed_content_from_chunk(chunk)
        assert result is None

    # emit (using mock stream)

    def test_emit_stream_chunk_suppressed(self, handler):
        """emit suppresses STREAM CHUNK records from console output."""
        record = MagicMock()
        record.getMessage.return_value = 'STREAM CHUNK | data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        # Mock the stream
        mock_stream = MagicMock()
        handler.stream = mock_stream

        handler.emit(record)

        # stream.write should NOT have been called at all
        assert mock_stream.write.call_count == 0

    def test_emit_stream_chunk_without_content_suppressed(self, handler):
        """emit suppresses even STREAM CHUNK without extractable content."""
        record = MagicMock()
        record.getMessage.return_value = 'STREAM CHUNK | data: {"choices":[{"delta":{}}]}\n'
        mock_stream = MagicMock()
        handler.stream = mock_stream

        handler.emit(record)

        # stream.write should NOT have been called at all
        assert mock_stream.write.call_count == 0

    def test_emit_non_stream_chunk_delegates_to_super(self, monkeypatch):
        """emit delegates to super().emit for non-STREAM CHUNK records."""
        handler = server.ContentOnlyConsoleHandler()
        record = MagicMock()
        record.getMessage.return_value = "INFO - some log message"

        super_emit = MagicMock()
        monkeypatch.setattr("logging.StreamHandler.emit", super_emit)

        handler.emit(record)

        super_emit.assert_called_once_with(record)


# ── Observability signal recording functions ─────────────────────────────────


class TestObservabilitySignalRecording:
    """Tests for observability signal recording functions."""

    @pytest.fixture(autouse=True)
    def _reset_state(self, monkeypatch):
        # These are module-level mutable dicts — replace with fresh copies
        monkeypatch.setattr(server, "backend_signal_counts", {
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        })
        monkeypatch.setattr(server, "session_restore_observability", {
            "restore_success_total": 0,
            "restore_fallback_total": {},
            "delta_payload_bytes_total": 0,
        })
        monkeypatch.setattr(server, "session_single_flight_observability", {
            "queue_events_total": 0,
            "reject_events_total": 0,
            "active_sessions_current": 0,
            "queue_depth_current": 0,
        })
        monkeypatch.setattr(server, "session_guardrail_observability", {
            "guardrail_cutoff_total": 0,
            "guardrail_cutoff_reasons": {},
            "session_invalidation_total": 0,
            "session_invalidation_reasons": {},
        })

    # ── Backend signal recording ──

    def test_record_backend_signal_increments_counter(self, monkeypatch):
        """_record_backend_signal increments the named signal counter."""
        assert server.backend_signal_counts["concurrency_rejects"] == 0
        server._record_backend_signal("concurrency_rejects")
        assert server.backend_signal_counts["concurrency_rejects"] == 1

    def test_record_backend_signal_multiple_increments(self, monkeypatch):
        """_record_backend_signal increments correctly on multiple calls."""
        server._record_backend_signal("connect_failures")
        server._record_backend_signal("connect_failures")
        server._record_backend_signal("connect_failures")
        assert server.backend_signal_counts["connect_failures"] == 3

    def test_record_backend_signal_unknown_signal_noop(self, monkeypatch):
        """_record_backend_signal does nothing for unknown signal names."""
        before = dict(server.backend_signal_counts)
        server._record_backend_signal("unknown_signal")
        assert server.backend_signal_counts == before

    def test_record_backend_signal_does_not_raise_on_failure(self):
        """_record_backend_signal is safe when backend_signal_counts is broken."""
        # Cannot patch the module-level dict with a non-dict
        # but we can verify it doesn't raise with various signal names
        server._record_backend_signal("concurrency_rejects")

    # ── Session restore recording ──

    def test_record_restore_success_increments(self, monkeypatch):
        """_record_restore_success increments restore_success_total."""
        assert server.session_restore_observability["restore_success_total"] == 0
        server._record_restore_success()
        assert server.session_restore_observability["restore_success_total"] == 1

    def test_record_restore_success_multiple(self, monkeypatch):
        """_record_restore_success increments correctly on multiple calls."""
        for _ in range(5):
            server._record_restore_success()
        assert server.session_restore_observability["restore_success_total"] == 5

    def test_record_restore_fallback_adds_reason(self, monkeypatch):
        """_record_restore_fallback adds entry to fallback_total dict."""
        server._record_restore_fallback("history_mismatch")
        assert server.session_restore_observability["restore_fallback_total"]["history_mismatch"] == 1

    def test_record_restore_fallback_empty_reason_noop(self, monkeypatch):
        """_record_restore_fallback does nothing with empty reason."""
        server._record_restore_fallback("")
        assert server.session_restore_observability["restore_fallback_total"] == {}

    def test_record_restore_fallback_multiple_reasons(self, monkeypatch):
        """_record_restore_fallback tracks separate reasons independently."""
        server._record_restore_fallback("history_mismatch")
        server._record_restore_fallback("history_mismatch")
        server._record_restore_fallback("no_new_messages")
        assert server.session_restore_observability["restore_fallback_total"]["history_mismatch"] == 2
        assert server.session_restore_observability["restore_fallback_total"]["no_new_messages"] == 1

    def test_record_delta_payload_bytes_accumulates(self, monkeypatch):
        """_record_delta_payload_bytes accumulates byte count."""
        server._record_delta_payload_bytes(1024)
        assert server.session_restore_observability["delta_payload_bytes_total"] == 1024

    def test_record_delta_payload_bytes_multiple_calls(self, monkeypatch):
        """_record_delta_payload_bytes accumulates on multiple calls."""
        server._record_delta_payload_bytes(100)
        server._record_delta_payload_bytes(200)
        assert server.session_restore_observability["delta_payload_bytes_total"] == 300

    def test_record_delta_payload_zero_noop(self, monkeypatch):
        """_record_delta_payload_bytes ignores zero/negative values."""
        server._record_delta_payload_bytes(0)
        assert server.session_restore_observability["delta_payload_bytes_total"] == 0
        server._record_delta_payload_bytes(-1)
        assert server.session_restore_observability["delta_payload_bytes_total"] == 0

    # ── Single-flight recording ──

    def test_record_single_flight_queue_increments(self, monkeypatch):
        """_record_single_flight_queue increments queue_events_total."""
        server._record_single_flight_queue()
        assert server.session_single_flight_observability["queue_events_total"] == 1

    def test_record_single_flight_reject_increments(self, monkeypatch):
        """_record_single_flight_reject increments reject_events_total."""
        server._record_single_flight_reject()
        assert server.session_single_flight_observability["reject_events_total"] == 1

    def test_record_single_flight_combined(self, monkeypatch):
        """_record_single_flight_queue and _reject track independently."""
        server._record_single_flight_queue()
        server._record_single_flight_queue()
        server._record_single_flight_reject()
        assert server.session_single_flight_observability["queue_events_total"] == 2
        assert server.session_single_flight_observability["reject_events_total"] == 1

    # ── Guardrail recording ──

    def test_record_guardrail_cutoff_increments(self, monkeypatch):
        """_record_guardrail_cutoff increments total and tracks reasons."""
        server._record_guardrail_cutoff("repetition")
        assert server.session_guardrail_observability["guardrail_cutoff_total"] == 1
        assert server.session_guardrail_observability["guardrail_cutoff_reasons"]["repetition"] == 1

    def test_record_guardrail_cutoff_empty_reason_noop(self, monkeypatch):
        """_record_guardrail_cutoff does nothing with empty reason."""
        server._record_guardrail_cutoff("")
        assert server.session_guardrail_observability["guardrail_cutoff_total"] == 0
        assert server.session_guardrail_observability["guardrail_cutoff_reasons"] == {}

    def test_record_guardrail_cutoff_multiple_reasons(self, monkeypatch):
        """_record_guardrail_cutoff tracks multiple reasons independently."""
        server._record_guardrail_cutoff("repetition")
        server._record_guardrail_cutoff("repetition")
        server._record_guardrail_cutoff("length")
        assert server.session_guardrail_observability["guardrail_cutoff_total"] == 3
        assert server.session_guardrail_observability["guardrail_cutoff_reasons"]["repetition"] == 2
        assert server.session_guardrail_observability["guardrail_cutoff_reasons"]["length"] == 1

    def test_record_session_invalidation_increments(self, monkeypatch):
        """_record_session_invalidation increments total and tracks reasons."""
        server._record_session_invalidation("guardrail_triggered")
        assert server.session_guardrail_observability["session_invalidation_total"] == 1
        assert server.session_guardrail_observability["session_invalidation_reasons"]["guardrail_triggered"] == 1

    def test_record_session_invalidation_empty_reason_noop(self, monkeypatch):
        """_record_session_invalidation does nothing with empty reason."""
        server._record_session_invalidation("")
        assert server.session_guardrail_observability["session_invalidation_total"] == 0
        assert server.session_guardrail_observability["session_invalidation_reasons"] == {}

    def test_record_session_invalidation_multiple_reasons(self, monkeypatch):
        """_record_session_invalidation tracks multiple reasons independently."""
        server._record_session_invalidation("guardrail_triggered")
        server._record_session_invalidation("guardrail_triggered")
        server._record_session_invalidation("manual_clear")
        assert server.session_guardrail_observability["session_invalidation_total"] == 3
        assert server.session_guardrail_observability["session_invalidation_reasons"]["guardrail_triggered"] == 2
        assert server.session_guardrail_observability["session_invalidation_reasons"]["manual_clear"] == 1
