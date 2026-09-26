"""Tests for the stable machine-readable exhaustion error code
(LP-0MU56ZM0K005F69G).

Contract:

- Every exhaustion 503 body carries ``error == "All providers exhausted"``
  (the string older clients already recognise).
- A stable ``code`` discriminates the paths: ``all_exhausted`` for generic
  exhaustion, ``outside_time_window`` for the sole-cause time-window case,
  and ``all_slots_exhausted`` for the local slot-exhaustion 429.
- The human-readable detail lives in ``detail``/``message`` and existing
  diagnostic fields (``retry_after``, ``unavailable_providers``,
  ``diagnostics``) are retained.
"""

import json
import time
from datetime import UTC, datetime
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response


class _FixedDateTime(datetime):
    fixed_now: datetime = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed_now


class _DummyRequest:
    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def _reset_provider_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


# ---------------------------------------------------------------------------
# Unit: response builders
# ---------------------------------------------------------------------------


class TestExhaustionCodes:
    def test_generic_503_has_stable_code(self):
        r = provider._build_exhausted_response(
            unavailable_providers={"a": 120},
            diagnostics=[{"provider": "a", "status": "cooldown"}],
        )
        body = json.loads(r.body)
        assert r.status_code == 503
        assert body["error"] == "All providers exhausted"
        assert body["code"] == "all_exhausted"
        assert body["retry_after"] == 120
        assert body["unavailable_providers"] == {"a": 120}
        assert body["diagnostics"] == [{"provider": "a", "status": "cooldown"}]
        assert r.headers.get("retry-after") == "120"

    def test_time_window_503_has_stable_code_and_detail(self):
        attempts = [
            {"provider": "w", "type": "remote", "status": "outside_time_window"}
        ]
        r = provider._build_time_window_exhausted_response(
            attempts, {}, False,
            model_config={
                "providers": [
                    {"name": "w", "available_times": ["09:00-17:00"]}
                ]
            },
        )
        assert r is not None
        body = json.loads(r.body)
        assert r.status_code == 503
        assert body["error"] == "All providers exhausted"
        assert body["code"] == "outside_time_window"
        assert "scheduled time window" in body["detail"]
        assert body["diagnostics"] == attempts

    def test_slot_exhaustion_429_has_stable_code(self):
        r = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=4
        )
        body = json.loads(r.body)
        assert r.status_code == 429
        assert body["error"] == "All providers exhausted"
        assert body["code"] == "all_slots_exhausted"
        assert "slots available" in body["message"]
        assert "0/4" in body["message"]

    def test_codes_are_module_constants(self):
        assert provider.EXHAUSTION_CODE_ALL == "all_exhausted"
        assert provider.EXHAUSTION_CODE_TIME_WINDOW == "outside_time_window"
        assert provider.EXHAUSTION_CODE_SLOTS == "all_slots_exhausted"


# ---------------------------------------------------------------------------
# Integration: fallback chain responses
# ---------------------------------------------------------------------------


async def _exhaust(model: dict) -> Response:
    async def _never_called(_req, _path, _cfg):  # pragma: no cover - guard
        raise AssertionError("upstream must not be contacted")

    with patch("proxy.server.proxy_to_remote", _never_called):
        return await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", model,
            {"provider_cooldown_seconds": 60},
        )


class TestFallbackExhaustionCodes:
    @pytest.mark.asyncio
    async def test_all_window_chain_emits_window_code(self):
        with patch("proxy.provider.datetime", _FixedDateTime):
            model = {
                "providers": [
                    {"name": "w1", "type": "remote",
                     "available_times": ["09:00-17:00"]},
                    {"name": "w2", "type": "remote",
                     "available_times": ["09:00-17:00"]},
                ]
            }
            result = await _exhaust(model)

        body = json.loads(result.body)
        assert body["error"] == "All providers exhausted"
        assert body["code"] == "outside_time_window"
        assert body["detail"]

    @pytest.mark.asyncio
    async def test_cooldown_only_chain_emits_generic_code(self):
        model = {"providers": [{"name": "c1", "type": "remote"}]}
        provider.mark_provider_unavailable("c1", 300.0)

        result = await _exhaust(model)

        body = json.loads(result.body)
        assert body["error"] == "All providers exhausted"
        assert body["code"] == "all_exhausted"
        assert body["unavailable_providers"]["c1"] >= 290

    @pytest.mark.asyncio
    async def test_quarantine_only_chain_emits_generic_code(self):
        model = {"providers": [{"name": "q1", "type": "remote"}]}
        provider._usage_reset_at[
            provider._usage_limit_account_key(model["providers"][0])
        ] = time.time() + 600

        result = await _exhaust(model)

        body = json.loads(result.body)
        assert body["code"] == "all_exhausted"
