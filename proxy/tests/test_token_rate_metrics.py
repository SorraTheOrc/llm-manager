"""Tests for token-rate Prometheus metrics (Gauge + Histogram).

These tests are written test-first: they assert the presence of the
metrics and their exposition on /metrics. The implementation is in
proxy/proxy/metrics.py and instrumentation will emit values during streaming.
"""

import importlib
import sys
from unittest.mock import patch

import pytest
import httpx

import proxy.metrics as metrics
from proxy import server as server_module


def test_token_rate_metrics_exist():
    """The token-rate gauge and histogram should be defined when prometheus_client available."""
    assert metrics._enabled, "prometheus_client must be available in test environment"
    assert hasattr(metrics, 'llama_token_rate_gauge'), "llama_token_rate_gauge should be defined"
    assert hasattr(metrics, 'llama_token_rate_histogram'), "llama_token_rate_histogram should be defined"


@pytest.mark.asyncio
async def test_metrics_endpoint_contains_token_rate_names(monkeypatch):
    """The /metrics endpoint should include the token-rate metric names."""
    # Ensure metrics module is loaded and configured
    # Use the proxy app and ASGI transport to get /metrics
    transport = httpx.ASGITransport(app=server_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        # Metric names should appear in the body when prometheus_client is available
        assert 'llama_token_rate_gauge' in body or 'llama_token_rate_histogram' in body, (
            "Expected token-rate metric names in /metrics output"
        )


def test_metrics_noop_when_prometheus_unavailable(monkeypatch):
    """When prometheus_client is not importable, the metrics module should be safe/no-op."""
    # Reload metrics module with prometheus_client removed
    with patch.dict(sys.modules, {"prometheus_client": None}):
        # Force reload to pick up fallback path
        try:
            import importlib
            import proxy.metrics as pm
            importlib.reload(pm)
            assert pm._enabled is False
            # Should not raise when calling generate_metrics_payload
            payload, ctype = pm.generate_metrics_payload()
            assert isinstance(payload, (bytes, bytearray))
        finally:
            importlib.reload(metrics)
