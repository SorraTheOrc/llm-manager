"""Tests for honest Retry-After (LP-0MT654ISW002QVH4).

AC1: retry_after reflects real availability (not hardcoded 60)
AC2: Retry-After header set on 503s with same value
AC3: max of cooldowns, window edges, usage resets
AC4: 0 when all providers available
"""

import json
import time
from datetime import UTC, datetime
from unittest.mock import patch

import proxy.provider as provider
from fastapi import Response


class TestComputeRetryAfter:
    def test_empty_returns_zero(self):
        assert provider._compute_retry_after() == 0
        assert provider._compute_retry_after({}) == 0
        assert provider._compute_retry_after(None) == 0

    def test_cooldown_max(self):
        assert provider._compute_retry_after({"a": 100, "b": 50}) == 100
        assert provider._compute_retry_after({"a": 30, "b": 200}) == 200

    def test_usage_reset_included(self):
        provider._usage_reset_at.clear()
        provider._usage_reset_at["acct"] = time.time() + 500
        try:
            result = provider._compute_retry_after({"a": 100})
            # 500 > 100 so usage reset wins
            assert 490 <= result <= 500
        finally:
            provider._usage_reset_at.clear()

    def test_usage_reset_max_with_cooldown(self):
        provider._usage_reset_at.clear()
        provider._usage_reset_at["acct"] = time.time() + 50
        try:
            result = provider._compute_retry_after({"a": 200})
            assert result == 200
        finally:
            provider._usage_reset_at.clear()

    def test_window_edge_included(self):
        now = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)
        cfg = {"name": "w", "available_times": ["09:00-17:00"]}
        # At 20:00, next window is 09:00 tomorrow = 13h = 46800s
        result = provider._compute_retry_after(
            model_config={"providers": [cfg]}, now_utc=now
        )
        assert result == 46800

    def test_max_across_all_sources(self):
        now = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)
        cfg = {"name": "w", "available_times": ["09:00-17:00"]}
        provider._usage_reset_at.clear()
        provider._usage_reset_at["acct"] = time.time() + 3600
        try:
            result = provider._compute_retry_after(
                unavailable_providers={"a": 100},
                model_config={"providers": [cfg]},
                now_utc=now,
            )
            assert result == 46800  # window wins over 3600 and 100
        finally:
            provider._usage_reset_at.clear()

    def test_expired_usage_reset_ignored(self):
        provider._usage_reset_at.clear()
        provider._usage_reset_at["acct"] = time.time() - 100  # already expired
        try:
            assert provider._compute_retry_after({"a": 10}) == 10
        finally:
            provider._usage_reset_at.clear()


class TestSecondsUntilNextWindow:
    def test_no_window_returns_none(self):
        assert provider._seconds_until_next_window({"name": "x"}) is None

    def test_inside_window_returns_none(self):
        cfg = {"name": "x", "available_times": ["09:00-17:00"]}
        now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        assert provider._seconds_until_next_window(cfg, now_utc=now) is None

    def test_outside_window_returns_seconds(self):
        cfg = {"name": "x", "available_times": ["09:00-17:00"]}
        now = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)
        result = provider._seconds_until_next_window(cfg, now_utc=now)
        assert result == 46800.0

    def test_overnight_window(self):
        cfg = {"name": "x", "available_times": ["22:00-02:00"]}
        # At 03:00, next window is 22:00 today = 19h
        now = datetime(2026, 1, 1, 3, 0, 0, tzinfo=UTC)
        result = provider._seconds_until_next_window(cfg, now_utc=now)
        assert result == 19 * 3600


class TestBuildExhaustedResponse:
    def test_retry_after_reflects_cooldown(self):
        r = provider._build_exhausted_response(
            unavailable_providers={"a": 120, "b": 60}, diagnostics=[]
        )
        body = json.loads(r.body)
        assert body["retry_after"] == 120
        assert r.headers.get("retry-after") == "120"

    def test_retry_after_zero_when_no_cooldown(self):
        r = provider._build_exhausted_response(
            unavailable_providers={}, diagnostics=[]
        )
        body = json.loads(r.body)
        assert body["retry_after"] == 0
        assert r.headers.get("retry-after") == "0"

    def test_retry_after_none_unavailable(self):
        r = provider._build_exhausted_response(diagnostics=[])
        body = json.loads(r.body)
        assert body["retry_after"] == 0

    def test_retry_after_header_present(self):
        r = provider._build_exhausted_response(
            unavailable_providers={"x": 45}, diagnostics=[]
        )
        # Starlette lowercases header keys
        assert "retry-after" in r.headers
        assert r.headers["retry-after"] == "45"

    def test_slot_exhaustion_no_retry_after(self):
        r = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=4
        )
        assert r.status_code == 429
        # 429 text/plain has no JSON retry_after
        assert b"slots available" in r.body

    def test_with_model_config_window(self):
        now = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)
        cfg = {"name": "w", "available_times": ["09:00-17:00"]}
        with patch("proxy.provider.datetime") as mock_dt:
            mock_dt.now.return_value = now
            r = provider._build_exhausted_response(
                unavailable_providers={},
                diagnostics=[],
                model_config={"providers": [cfg]},
            )
        body = json.loads(r.body)
        # Should reflect window edge, not 0
        assert body["retry_after"] == 46800


class TestBuildTimeWindowExhaustedResponse:
    def test_returns_retry_after_with_model_config(self):
        now = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)
        cfg = {"name": "w", "available_times": ["09:00-17:00"], "type": "remote"}
        attempts = [{"provider": "w", "status": "outside_time_window"}]
        with patch("proxy.provider.datetime") as mock_dt:
            mock_dt.now.return_value = now
            r = provider._build_time_window_exhausted_response(
                attempts, {}, False, model_config={"providers": [cfg]}
            )
        assert r is not None
        body = json.loads(r.body)
        assert body["retry_after"] == 46800
        assert r.headers.get("retry-after") == "46800"

    def test_returns_none_when_unavailable(self):
        attempts = [{"provider": "w", "status": "outside_time_window"}]
        r = provider._build_time_window_exhausted_response(
            attempts, {"a": 10}, False
        )
        assert r is None

    def test_returns_none_when_tried(self):
        attempts = [{"provider": "w", "status": "outside_time_window"}]
        r = provider._build_time_window_exhausted_response(
            attempts, {}, True
        )
        assert r is None
