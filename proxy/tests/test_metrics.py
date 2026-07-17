"""Tests for proxy metrics module (proxy/proxy/metrics.py).

Covers:
- proxy_http_errors_total counter definition and labels
- record_http_error() function behavior
- Best-effort no-op fallback when prometheus_client is unavailable
"""

import sys
from unittest.mock import patch


import proxy.metrics as metrics


class TestHttpErrorsCounter:
    """Tests for the proxy_http_errors_total counter and record_http_error()."""

    def test_http_errors_counter_exists(self):
        """The proxy_http_errors_total counter is defined when prometheus_client is available."""
        assert metrics._enabled, (
            "prometheus_client should be available; if not, check test environment"
        )
        assert metrics.proxy_http_errors_total is not None
        # Verify it's a Counter
        from prometheus_client import Counter
        assert isinstance(metrics.proxy_http_errors_total, Counter)

    def test_http_errors_counter_labels(self):
        """The counter has the expected labels: endpoint, status, reason."""
        # The label names are accessible via the ._labelnames attribute on Counter objects
        label_names = metrics.proxy_http_errors_total._labelnames
        assert "endpoint" in label_names
        assert "status" in label_names
        assert "reason" in label_names

    def test_record_http_error_increments_counter(self):
        """Calling record_http_error() increments the counter with matching labels."""
        # Get value before
        before = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()

        metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")

        after = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()

        assert after == before + 1, "counter should increment by 1"

    def test_record_http_error_separate_reasons(self):
        """Different reason labels produce separate label combinations (no cross-talk)."""
        before_backend = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()
        before_unavail = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_unavailable",
        )._value.get()

        metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")

        after_backend = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()
        after_unavail = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_unavailable",
        )._value.get()

        assert after_backend == before_backend + 1
        assert after_unavail == before_unavail, "unrelated reason should not be affected"

    def test_record_http_error_different_endpoint(self):
        """Different endpoint labels are tracked independently."""
        before_completions = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()
        before_other = metrics.proxy_http_errors_total.labels(
            endpoint="v1/other",
            status="5xx",
            reason="backend_error",
        )._value.get()

        metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")

        after_other = metrics.proxy_http_errors_total.labels(
            endpoint="v1/other",
            status="5xx",
            reason="backend_error",
        )._value.get()

        assert after_other == before_other, "other endpoint should not be affected"

    def test_record_http_error_different_status(self):
        """Different status labels are tracked independently."""
        before_5xx = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="5xx",
            reason="backend_error",
        )._value.get()
        before_4xx = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="4xx",
            reason="backend_error",
        )._value.get()

        metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")

        after_4xx = metrics.proxy_http_errors_total.labels(
            endpoint="v1/chat/completions",
            status="4xx",
            reason="backend_error",
        )._value.get()

        assert after_4xx == before_4xx, "4xx status should not be affected"


class TestHttpErrorsCounterBestEffort:
    """Verify best-effort fallback when prometheus_client is not available."""

    def test_record_http_error_noop_when_disabled(self):
        """record_http_error is a no-op (does not raise) when prometheus_client is disabled."""
        # Reload metrics module with prometheus_client unavailable
        with patch.dict(sys.modules, {"prometheus_client": None}):
            # Remove cached import
            with patch.object(metrics, "_enabled", False):
                # Should not raise
                metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")

    def test_record_http_error_safe_when_counter_none(self):
        """record_http_error is safe when proxy_http_errors_total is None."""
        with patch.object(metrics, "proxy_http_errors_total", None):
            metrics.record_http_error("v1/chat/completions", "5xx", "backend_error")
