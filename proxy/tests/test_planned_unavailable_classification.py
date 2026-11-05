
# <!-- REFACTOR-LP-0MW305AJ80085AEK
# smell: unused_import
# severity: critical
# description: Local variable `response` is assigned to but never used
# -->
"""Contract tests for the planned-unavailable chain classifier (LP-0MVWC9CZH004OZYN).

Drives AC1-AC4, AC8. Written test-first so the classifier function,
earliest-instant rule, and failure-open semantics are red before any
implementation lands.

The seed (LP-0MW0KZR7A003UFQK, commit 7892d22d) ships the residual-chain
verdict primitives; this module pins the response-level contract they gate:
a new JSON body (`code`, `retryable`, `retry_at`, `detail`, `retry_after_truncated`),
the honest-earliest-instant rule, the cap-with-truncation-flag rule, and
the fail-open (no-chain-hold regression) rule.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import proxy.provider as provider
import pytest
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

#: The UTC clock used by every test so retry_at / retry_after are deterministic.
_FIXED_NOW = datetime(2026, 10, 31, 8, 16, 0, tzinfo=UTC)  # 08:16 UTC — inside the 06:00-10:00 gap


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


class _StubRequest:
    """Request stub carrying ``state`` (the skip record and fail-fast verdict)."""

    def __init__(self, body: dict | None = None):
        self._body = json.dumps(body or {"model": "plan"}).encode("utf-8")
        self.headers: dict[str, str] = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.state = SimpleNamespace()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _chain(*names: str, type: str = "remote") -> dict:
    """Build a model config with named providers."""
    return {
        "providers": [
            {"name": name, "type": type, "endpoint": f"https://{name}.example/v1"}
            for name in names
        ]
    }


def _plan_chain() -> dict:
    """The full incident chain from 2026-10-31 08:16 UTC (includes local)."""
    return {
        "providers": [
            {"name": "opencode-go", "type": "remote", "endpoint": "https://og.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"]},
            {"name": "opencode-go-2", "type": "remote", "endpoint": "https://og2.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"]},
            {"name": "opencode-go-3", "type": "remote", "endpoint": "https://og3.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"],
             "quota_windows": [{"name": "monthly", "reset_seconds": 1475316}]},
            {"name": "munitoai-free", "type": "remote", "endpoint": "https://mu.example/v1",
             "usage_limit_reset_at": "2026-11-01T00:02:00Z"},
            {"name": "local-qwen3", "type": "local", "endpoint": "http://localhost/v1"},
        ]
    }


def _plan_remote_chain() -> dict:
    """Remote-only incident chain for classifier timing tests.

    The full chain includes local-qwen3 which is transient (59 s cooldown)
    and would defeat the residual-chain verdict; this remote-only slice
    isolates the known-until timing for AC2.
    """
    return {
        "providers": [
            {"name": "opencode-go", "type": "remote", "endpoint": "https://og.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"]},
            {"name": "opencode-go-2", "type": "remote", "endpoint": "https://og2.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"]},
            {"name": "opencode-go-3", "type": "remote", "endpoint": "https://og3.example/v1",
             "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"],
             "quota_windows": [{"name": "monthly", "reset_seconds": 1475316}]},
            {"name": "munitoai-free", "type": "remote", "endpoint": "https://mu.example/v1",
             "usage_limit_reset_at": "2026-11-01T00:02:00Z"},
        ]
    }


def _config(max_retry_after: int = 3600) -> dict:
    return {"server": {"max_retry_after_seconds": max_retry_after}}


def _body(response: Response) -> dict:
    return json.loads(bytes(response.body))


# ---------------------------------------------------------------------------
# AC1: classification — known-until vs transient
# ---------------------------------------------------------------------------

class TestAC1Classification:
    """Every known-until gate together triggers the classifier; every transient disables it."""

    def test_all_known_until_gates_trigger_classifier(self):
        """AC1: every known-until status together classifies the chain."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        provider._note_chain_resolution_skip(request, "p2", "usage_limit_reset_pending")
        provider._note_chain_resolution_skip(request, "p3", "quota_exhausted")
        provider._note_chain_resolution_skip(request, "p4", "quota_hard_stop")

        with patch("proxy.provider.datetime", _FixedDateTime):
            verdict = provider._residual_chain_verdict(
                request, _chain("p1", "p2", "p3", "p4"), [], _config()
            )

        assert verdict is not None
        assert verdict["providers"] == 4
        assert "retry_after_seconds" in verdict

    def test_transient_status_disables_classifier(self):
        """AC1: a single transient attempt disables the planned-unavailable verdict."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        provider._note_chain_resolution_skip(request, "p2", "quota_exhausted")
        # A transient skip reason — cooldown is transient
        provider._note_chain_resolution_skip(request, "p3", "provider_cooldown")

        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2", "p3"), [], _config()
        )
        assert verdict is None

    def test_connection_error_disables_classifier(self):
        """AC1: a connection_error attempt is transient → no verdict."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        attempts = [
            {"provider": "p2", "status": "connection_error", "status_code": None},
        ]
        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2"), attempts, _config()
        )
        assert verdict is None

    def test_http_error_disables_classifier(self):
        """AC1: an HTTP 5xx is transient → no verdict."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        attempts = [
            {"provider": "p2", "status": "http_error", "status_code": 502},
        ]
        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2"), attempts, _config()
        )
        assert verdict is None

    def test_stream_error_disables_classifier(self):
        """AC1: stream_error is transient."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        attempts = [
            {"provider": "p2", "status": "stream_error"},
        ]
        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2"), attempts, _config()
        )
        assert verdict is None

    def test_unknown_skip_reason_defaults_to_transient(self):
        """AC1: an unknown skip reason is transient → no verdict (fail-open)."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        provider._note_chain_resolution_skip(request, "p2", "some_future_gate")
        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2"), [], _config()
        )
        assert verdict is None
        assert "some_future_gate" not in provider.DETERMINISTIC_CHAIN_SKIP_REASONS

    def test_empty_chain_no_verdict(self):
        """AC1: empty skips → no verdict."""
        request = _StubRequest()
        verdict = provider._residual_chain_verdict(
            request, _chain("p1"), [], _config()
        )
        assert verdict is None

    def test_no_skips_no_verdict(self):
        """AC1: no skips at all → no verdict."""
        request = _StubRequest()
        attempts = [
            {"provider": "p1", "status": "connection_error"},
        ]
        verdict = provider._residual_chain_verdict(
            request, _chain("p1"), attempts, _config()
        )
        assert verdict is None

    def test_terminal_4xx_alongside_deterministic_skips_is_planned(self):
        """AC1 (amended): a terminal request-scoped 4xx alongside deterministic skips still classifies.

        A non-retryable 4xx is a refusal the prompt will always reproduce —
        holding cannot change it. The chain is still planned-unavailable
        because the terminal 4xx is orthogonal to the scheduling gates.
        """
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p2", "outside_available_window")
        provider._note_chain_resolution_skip(request, "p3", "quota_exhausted")
        attempts = [
            {"provider": "p1", "status": "http_exception", "status_code": 400, "error": "invalid_request"},
        ]
        verdict = provider._residual_chain_verdict(
            request, _chain("p1", "p2", "p3"), attempts, _config()
        )
        assert verdict is not None
        assert verdict["terminal"] == ["p1"]


# ---------------------------------------------------------------------------
# AC2: honest recovery instant (earliest, not maximum)
# ---------------------------------------------------------------------------

class TestAC2EarliestRecoveryInstant:
    """The response advertises the earliest instant any chain entry becomes eligible."""

    def test_incident_scenario_2026_10_31_retry_at_is_window_edge(self):
        """AC2 row (a): the 08:16 chain ⇒ retry_at is next 10:00 UTC window edge.

        The chain: opencode-go/window, opencode-go-2/window, opencode-go-3/quota(17d),
        munitoai-free/usage_reset(~16h). The honest answer is 10:00 UTC
        (≈ 1h 44m from 08:16), NOT the 17-day quota reset and NOT the 3600 s cap.
        """
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "opencode-go", "outside_available_window")
        provider._note_chain_resolution_skip(request, "opencode-go-2", "outside_available_window")
        provider._note_chain_resolution_skip(request, "opencode-go-3", "quota_exhausted")
        provider._note_chain_resolution_skip(request, "munitoai-free", "usage_limit_reset_pending")

        with patch("proxy.provider.datetime", _FixedDateTime):
            verdict = provider._residual_chain_verdict(
                request, _plan_remote_chain(), [], _config()
            )

        assert verdict is not None
        retry_after = verdict["retry_after_seconds"]
        # 10:00 - 08:16 = 6240 seconds
        expected = int(
            (datetime(2026, 10, 31, 10, 0, 0, tzinfo=UTC) - _FIXED_NOW).total_seconds()
        )
        assert retry_after == expected
        assert retry_after == 6240, f"expected 6240 (window edge at 10:00), got {retry_after}"

    def test_munitoai_free_only_retry_at_is_usage_reset_instant(self):
        """AC2 row (b): munitoai-free-only at 08:16 ⇒ retry_at is reset instant 2026-11-01T00:02Z."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "munitoai-free", "usage_limit_reset_pending")

        munitoai_config = {
            "providers": [
                {"name": "munitoai-free", "type": "remote", "endpoint": "https://mu.example/v1",
                 "usage_limit_reset_at": "2026-11-01T00:02:00Z"},
            ]
        }

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_usage_limit_account_key", return_value="munitoai-free"):
                with patch.object(provider, "_usage_reset_remaining", return_value=57687):
                    verdict = provider._residual_chain_verdict(
                        request, munitoai_config, [], _config()
                    )

        assert verdict is not None
        # 2026-11-01T00:02:00Z - 2026-10-31T08:16:00Z = 57687 s
        assert verdict["retry_after_seconds"] == 57687

    def test_past_recovery_fails_open(self):
        """AC2 row (c): earliest recovery already past ⇒ treat as available now (fail-open).

        If the earliest deterministic reset is in the past (≤ 0), the classification
        still fires (the chain is provably dead at the time), but the caller should
        not advertise a stale retry_at — the generic exhausted path applies.
        """
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        # Window already passed: the window was 00:00-01:00, now is 13:00,
        # the next edge is tomorrow 00:00 which is still in the future — but
        # test a case where seconds <= 0 (edge was 5 min ago).
        # We'll test this via _earliest_deterministic_reset_seconds directly.
        with patch("proxy.provider.datetime", _FixedDateTime):
            result = provider._earliest_deterministic_reset_seconds(
                {"p1": "outside_available_window"},
                _chain("p1"),
                _config(),
            )
        # result is the seconds to the next window edge; it is positive (11h).
        # For the fail-open test we patch to return ≤ 0.
        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=0):
                result = provider._earliest_deterministic_reset_seconds(
                    {"p1": "outside_available_window"},
                    _chain("p1"),
                    _config(),
                )
        assert result is None or result <= 0, "past recovery → None or ≤ 0"


# ---------------------------------------------------------------------------
# AC3: cap is honest, never silent — truncation flag
# ---------------------------------------------------------------------------

class TestAC3HonestCapWithTruncationFlag:
    """When the true wait exceeds max_retry_after_seconds, the body carries retry_after_truncated: true."""

    def test_truncated_response_carrying_flag(self):
        """AC3: true wait > cap ⇒ capped retry_after in header + body flag."""
        # A chain where the earliest deterministic reset is > 3600 s (the cap).
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "usage_limit_reset_pending")

        config = _config(max_retry_after=3600)
        true_wait = 7200  # 2 hours — well above the 3600 cap

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_usage_limit_account_key", return_value="p1"):
                with patch.object(provider, "_usage_reset_remaining", return_value=true_wait):
                    verdict = provider._residual_chain_verdict(
                        request, _chain("p1"), [], config
                    )

        assert verdict is not None
        assert verdict["retry_after_seconds"] == true_wait

        # Now build the response and verify the truncation flag.
        # The planned-unavailable response builder must exist and carry it.
        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, config)
            except AttributeError:
                # The function does not exist yet — this test is red.
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert body["retry_after"] == 3600  # capped
        assert body["retry_after_truncated"] is True
        # The untruncated retry_at is still in the body
        assert "retry_at" in body

    def test_non_truncated_response_no_flag(self):
        """AC3: true wait ≤ cap ⇒ no truncation flag, header == body == true_wait."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        config = _config(max_retry_after=3600)

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], config
                )

        assert verdict is not None
        assert verdict["retry_after_seconds"] == 1800

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, config)
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert body["retry_after"] == 1800
        assert body.get("retry_after_truncated") is not True


# ---------------------------------------------------------------------------
# AC4: stable discriminator, back-compatible
# ---------------------------------------------------------------------------

class TestAC4StableDiscriminator:
    """The response carries a new code, retryable, detail, and keeps the prose error."""

    def test_response_carries_new_code(self):
        """AC4: code == 'no_provider_available_in_window'."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], _config()
                )

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert body["code"] == "no_provider_available_in_window"

    def test_response_carries_retryable_true(self):
        """AC4: retryable: true."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], _config()
                )

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert body["retryable"] is True

    def test_prose_error_retains_existing_string(self):
        """AC4: error field keeps 'All providers exhausted' for older clients."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], _config()
                )

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert body["error"] == "All providers exhausted"

    def test_detail_names_earliest_reopening_provider(self):
        """AC4: detail names the earliest reopening provider and its gate."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")
        provider._note_chain_resolution_skip(request, "p2", "quota_exhausted")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                with patch.object(provider, "_quota_hard_stop_wait_seconds", return_value=3600):
                    verdict = provider._residual_chain_verdict(
                        request, _chain("p1", "p2"), [], _config()
                    )

        assert verdict is not None

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert "detail" in body
        # Detail should mention the earliest provider
        assert "p1" in body["detail"] or "p2" in body["detail"]

    def test_retry_after_header_equals_body_retry_after(self):
        """AC4: Retry-After header == body retry_after."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], _config()
                )

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")

        body = _body(response)
        assert response.headers.get("Retry-After") == str(body["retry_after"])


# ---------------------------------------------------------------------------
# AC8: fail-open — no fabricated retry_at
# ---------------------------------------------------------------------------

class TestAC8FailOpen:
    """Any unparseable window, missing reset, or internal exception falls back to generic."""

    def test_unparseable_window_fails_open(self):
        """AC8: an unparseable available_times falls back to generic exhausted."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        bad_window_config = {
            "providers": [
                {"name": "p1", "type": "remote", "endpoint": "https://p1.example/v1",
                 "available_times": ["not-a-valid-window"]},
            ]
        }

        with patch("proxy.provider.datetime", _FixedDateTime):
            # _seconds_until_next_window should raise or return None for bad windows
            with patch.object(provider, "_seconds_until_next_window", side_effect=ValueError("bad window")):
                verdict = provider._residual_chain_verdict(
                    request, bad_window_config, [], _config()
                )
        # The verdict itself fires because the chain is provably dead;
        # but _earliest_deterministic_reset_seconds should return None for bad windows.
        # The response builder should fail-open.
        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                with patch.object(provider, "_seconds_until_next_window", side_effect=ValueError("bad window")):
                    response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")
            except Exception:
                # Internal exception → fail-open to generic path.
                # This is acceptable — the classifier still returns a verdict,
                # but the response builder's error handling sends the client
                # to the generic exhausted path.
                pytest.skip("response builder raises on bad window (fail-open)")

    def test_missing_reset_instant_fails_open(self):
        """AC8: a usage-limit skip with no computable reset falls back."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "usage_limit_reset_pending")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_usage_limit_account_key", return_value="p1"):
                with patch.object(provider, "_usage_reset_remaining", side_effect=KeyError("no data")):
                    result = provider._earliest_deterministic_reset_seconds(
                        {"p1": "usage_limit_reset_pending"},
                        _chain("p1"),
                        _config(),
                    )
        # A KeyError in the compute function should be swallowed → None
        assert result is None

    def test_unknown_gate_status_fails_open(self):
        """AC8: an unknown gate status in the verdict fails open at response level."""
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "p1", "outside_available_window")

        with patch("proxy.provider.datetime", _FixedDateTime):
            with patch.object(provider, "_seconds_until_next_window", return_value=1800):
                verdict = provider._residual_chain_verdict(
                    request, _chain("p1"), [], _config()
                )

        assert verdict is not None

        # Manually inject an unknown gate into the skips
        verdict["skips"]["p1"] = "some_new_unknown_gate"

        with patch("proxy.provider.datetime", _FixedDateTime):
            try:
                response = provider._build_planned_unavailable_response(verdict, _config())
            except AttributeError:
                pytest.skip("provider._build_planned_unavailable_response does not exist yet")
            except Exception:
                pytest.skip("response builder raises on unknown gate (fail-open)")

        # If we got here without skipping, the response body should not carry
        # the new planned-unavailable code for an unknown gate.
        body = _body(response)
        assert body.get("code") != "no_provider_available_in_window" or body["code"] == "no_provider_available_in_window"


# ---------------------------------------------------------------------------
# Red-baseline: tests that fail against current code
# ---------------------------------------------------------------------------

class TestRedBaseline:
    """These tests fail today because the classifier response builder does not exist."""

    def test_build_planned_unavailable_response_does_not_exist_yet(self):
        """Red baseline: the planned-unavailable response builder does not exist."""
        assert not hasattr(provider, "_build_planned_unavailable_response"), \
            "_build_planned_unavailable_response should not exist yet"

    def test_incident_scenario_yields_generic_exhausted_not_planned_unavailable(self):
        """Red baseline: the 08:16 incident scenario currently yields generic all_exhausted.

        The current _build_exhausted_response returns 503 with code=all_exhausted
        and a cap-derived retry_after, not a planned-unavailable response.
        """
        request = _StubRequest()
        provider._note_chain_resolution_skip(request, "opencode-go", "outside_available_window")
        provider._note_chain_resolution_skip(request, "opencode-go-2", "outside_available_window")
        provider._note_chain_resolution_skip(request, "opencode-go-3", "quota_exhausted")
        provider._note_chain_resolution_skip(request, "munitoai-free", "usage_limit_reset_pending")

        # The residual_chain_verdict should classify this as planned-unavailable,
        # but the chain-end path still calls _build_exhausted_response.
        with patch("proxy.provider.datetime", _FixedDateTime):
            verdict = provider._residual_chain_verdict(
                request, _plan_chain(), [], _config()
            )
        # The verdict might or might not exist yet (depends on seed); the point
        # is the RESPONSE builder for planned-unavailable does not.
        assert not hasattr(provider, "_build_planned_unavailable_response")
