
# <!-- REFACTOR-LP-0MVB0THT7000KT0G
# smell: unused_import
# severity: critical
# description: Local variable `config` is assigned to but never used
# -->
"""
Grace-period immunity for local cold-start 503s and connection errors.

Parent: LP-0MV9UNIOZ009RIAN (Implement grace-period immunity: local cold-start
503s must not trigger a 60s local cooldown)

Context
-------
After a proxy restart, requests that race llama-server warm-up receive HTTP 503
(or connection errors) and flow into ``_handle_http_error_with_cooldown()`` /
``_handle_connection_error_in_fallback()`` in ``proxy/provider.py``, which call
``mark_provider_unavailable()`` — a ~60s local quarantine.  This feature adds
**grace-period immunity**: while ``cold_start.is_cold(config)`` is True, a local
HTTP 503 or connection error defers (Retry-After, no quarantine, no failure-
counter bump) instead of quarantining the provider.  After the window, or for
any other 5xx, the normal quarantine still applies so genuine faults are not
masked.

Test strategy
-------------
Write the tests first.  A red test reproduces the current quarantine behaviour;
after the implementation feature lands, the same scenarios are verified green.

Each test uses ``_reset_provider_state()`` to clear ``_provider_unavailable_until``
and ``_provider_failure_count``.  The ``_server_config()`` helper wraps server
config dicts for ``cold_start`` API calls.

Deterministic cold-start control
--------------------------------
Tests arm the cold window via ``cold_start.note_model_loaded()`` and manipulate
``cold_start._state.loaded_at`` to simulate elapsed time.  The ``autouse``
fixture resets cold-start state around every test.

Acceptance Criteria covered
---------------------------
(a) Cold-start 503 within ``is_cold()`` → defer via Retry-After, no cooldown,
    no failure-counter bump.
(b) Sustained HTTP 503 **after** the warm-up window → normal cooldown applies.
(c) Connection error during warm-up → defer, no cooldown.
(d) Concurrent cold-start dispatch gating still respected (existing tests in
    ``test_cold_start_admission.py`` cover this).
(e) A fresh warm-up window begins after each restart (new ``note_model_loaded()``
    arms it).
Non-503 5xx during warm-up still quarantines (genuine faults not masked).
Failure counter not polluted during the window.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from proxy import cold_start, provider

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_cold_start_and_provider():
    """Reset cold-start state and provider failure/cooldown state around every test."""
    cold_start.reset()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    cold_start.reset()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


def _server_config(**overrides) -> dict:
    """Build a minimal server config dict, merging overrides."""
    cfg = {"session_slot_pool_size": 3}
    cfg.update(overrides)
    return {"server": cfg}


def _local_config(**overrides) -> dict:
    """Build a config with local provider settings for cold-start checks.

    Defaults may be overridden (overrides win).
    """
    defaults = {
        "local_cold_start_grace_seconds": 120,
        "local_cold_start_max_concurrent": 1,
        "local_cold_start_retry_after_seconds": 15,
    }
    defaults.update(overrides)
    return _server_config(**defaults)


def _make_http_response(status_code: int, extra_headers: dict | None = None) -> MagicMock:
    """Build a MagicMock response for the provider error-handler primitives."""
    resp = MagicMock()
    resp.status_code = status_code
    headers = dict(extra_headers or {})
    resp.headers = headers
    return resp


# ── AC(a): Cold-start 503 within the warm-up window defers ──────────────────

class TestColdStart503Deferral:
    """AC(a): Cold-start 503 within is_cold() window → defer via Retry-After."""

    def test_cold_start_503_no_cooldown_local(self):
        """AC(a): Local HTTP 503 during cold window → no cooldown, no failure bump."""
        config = _local_config()
        cold_start.note_model_loaded()
        assert cold_start.is_cold(config), "Cold window must be armed for this test"

        attempts = []
        response = _make_http_response(503)
        cooldown = provider._handle_http_error_with_cooldown(
            response,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="Service Unavailable — model still loading",
        )

        # Immunity: no quarantine applied; the return value carries the
        # cold-start retry hint (Retry-After-seconds from config, default 15).
        assert cooldown == 15, (
            f"Cold-start 503 must return the retry hint, got {cooldown}"
        )
        assert not provider._is_provider_unavailable("local-llama"), (
            "Cold-start 503 must not mark provider unavailable"
        )
        assert provider._provider_failure_count.get("local-llama", 0) == 0, (
            "Cold-start 503 must not bump failure counter"
        )
        assert len(attempts) == 1
        assert attempts[0]["status"] == "cold_start_deferral"
        assert attempts[0].get("retry_after_seconds") == 15

    def test_cold_start_503_returns_retry_after(self):
        """AC(a): Cold-start 503 records retry_after_seconds in attempts."""
        config = _local_config(local_cold_start_retry_after_seconds=20)
        cold_start.note_model_loaded()

        attempts = []
        response = _make_http_response(503)
        cooldown = provider._handle_http_error_with_cooldown(
            response,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="Service Unavailable",
            config=config,
        )

        assert attempts[0]["status"] == "cold_start_deferral"
        assert attempts[0].get("retry_after_seconds") == 20
        # Return value is the retry hint.
        assert cooldown == 20

    def test_cold_start_503_remote_still_quarantines(self):
        """Remote provider 503 during cold window still quarantines normally."""
        cold_start.note_model_loaded()

        attempts = []
        response = _make_http_response(503)
        provider._handle_http_error_with_cooldown(
            response,
            "remote-primary",
            "remote",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="Service Unavailable",
        )

        # Remote providers are not immune — they quarantine as normal.
        assert provider._is_provider_unavailable("remote-primary"), (
            "Remote 503 during cold window must still quarantine"
        )
        assert provider._provider_failure_count.get("remote-primary", 0) >= 1, (
            "Remote provider failure count must still bump"
        )


# ── AC(b): Sustained 503 after warm-up window applies normal cooldown ────────

class TestPostWindowCooldown:
    """AC(b): Sustained HTTP 503 after the warm-up window → normal cooldown."""

    def test_503_after_grace_expires_applies_normal_cooldown(self):
        """AC(b): 503 received after is_cold() expires → normal quarantine."""
        config = _local_config(local_cold_start_grace_seconds=10)
        cold_start.note_model_loaded()
        loaded_at = cold_start._state.loaded_at

        # Wait for grace window to expire (simulated).
        assert cold_start.is_cold(config, now=loaded_at + 9)
        assert not cold_start.is_cold(config, now=loaded_at + 11)

        # Move the arm time into the real past so the handler's own is_cold()
        # (real monotonic clock) sees an expired window.
        cold_start._state.loaded_at = time.monotonic() - 15

        attempts = []
        response = _make_http_response(503)
        cooldown = provider._handle_http_error_with_cooldown(
            response,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="Service Unavailable",
            config=config,
        )

        # After window: normal quarantine applies.
        assert cooldown == 60.0, (
            "Post-window 503 must apply normal cooldown"
        )
        assert provider._is_provider_unavailable("local-llama"), (
            "Post-window 503 must mark provider unavailable"
        )
        # The attempt is a normal http_error quarantine, not a deferral.
        assert len(attempts) == 1
        assert attempts[0]["status"] == "http_error"

    def test_502_during_cold_still_quarantines(self):
        """Non-503 5xx during cold window still quarantines (no masking)."""
        config = _local_config()
        cold_start.note_model_loaded()

        attempts = []
        response = _make_http_response(502)
        cooldown = provider._handle_http_error_with_cooldown(
            response,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="Bad Gateway",
            config=config,
        )

        # Only 503 is immune during cold window; 502 still quarantines.
        assert cooldown == 60.0, (
            "Non-503 5xx during cold window must still quarantine"
        )
        assert provider._is_provider_unavailable("local-llama"), (
            "502 during cold window must mark provider unavailable"
        )


# ── AC(c): Connection error during warm-up defers ────────────────────────────

class TestColdStartConnectionError:
    """AC(c): Connection error during warm-up → defer, no cooldown."""

    def test_cold_start_connection_error_no_cooldown(self):
        """AC(c): Connection error during cold window → defer, no cooldown."""
        config = _local_config()
        cold_start.note_model_loaded()
        assert cold_start.is_cold(config)

        attempts = []
        exc = httpx.ConnectError("Connection refused")
        result = provider._handle_connection_error_in_fallback(
            exc,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            config=config,
        )

        assert result is True, "Connection error must return True (continue)"
        assert not provider._is_provider_unavailable("local-llama"), (
            "Cold-start connection error must not mark provider unavailable"
        )
        assert provider._provider_failure_count.get("local-llama", 0) == 0, (
            "Cold-start connection error must not bump failure counter"
        )
        assert len(attempts) == 1
        assert attempts[0]["status"] == "cold_start_deferral"

    def test_cold_start_connection_error_remote_still_quarantines(self):
        """Remote connection error during cold window still quarantines."""
        config = _local_config()
        cold_start.note_model_loaded()

        attempts = []
        exc = httpx.ConnectError("Connection refused")
        result = provider._handle_connection_error_in_fallback(
            exc,
            "remote-primary",
            "remote",
            cooldown_seconds=60.0,
            attempts=attempts,
            config=config,
        )

        assert result is True
        assert provider._is_provider_unavailable("remote-primary"), (
            "Remote connection error during cold window must still quarantine"
        )

    def test_connection_error_after_grace_expires(self):
        """Connection error after warm-up → normal quarantine."""
        config = _local_config(local_cold_start_grace_seconds=10)
        cold_start.note_model_loaded()

        # Simulate time passing past the grace window (loaded_at in the past,
        # relative to the real monotonic clock the handler uses).
        cold_start._state.loaded_at = time.monotonic() - 15

        attempts = []
        exc = httpx.ConnectError("Connection refused")
        result = provider._handle_connection_error_in_fallback(
            exc,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            config=config,
        )

        assert result is True
        assert provider._is_provider_unavailable("local-llama"), (
            "Post-window connection error must mark provider unavailable"
        )
        # Normal connection_error quarantine recorded, not a deferral.
        assert len(attempts) == 1
        assert attempts[0]["status"] == "connection_error"

    def test_non_connection_error_during_cold_still_quarantines(self):
        """Non-connection error during cold window returns False (no deferral)."""
        config = _local_config()
        cold_start.note_model_loaded()

        attempts = []
        exc = RuntimeError("Something unexpected")
        result = provider._handle_connection_error_in_fallback(
            exc,
            "local-llama",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            config=config,
        )

        assert result is False, (
            "Non-connection error must return False (caller handles it)"
        )
        assert len(attempts) == 0, (
            "Non-connection error must not record any attempt"
        )


# ── AC(e): Fresh warm-up window after restart ────────────────────────────────

class TestFreshWindowAfterRestart:
    """AC(e): A new note_model_loaded() arms a fresh warm-up window."""

    def test_fresh_window_rearms_immunity(self):
        """AC(e): After a model reload, immunity applies again from the new window."""
        config = _local_config(local_cold_start_grace_seconds=10)

        # First load → cold window armed.
        cold_start.note_model_loaded()
        assert cold_start.is_cold(config)

        attempts_1 = []
        response = MagicMock()
        response.status_code = 503
        response.headers = {}
        cooldown1 = provider._handle_http_error_with_cooldown(
            response, "local-llama", "local", 60.0, attempts_1, "cold",
            config=config,
        )
        assert cooldown1 == 15, "First load: cold-start 503 returns the retry hint"

        # Wait for window to expire (loaded_at in the past).
        cold_start._state.loaded_at = cold_start._state.loaded_at - 15
        assert not cold_start.is_cold(config)

        # 503 after window → normal quarantine.
        attempts_2 = []
        cooldown2 = provider._handle_http_error_with_cooldown(
            response, "local-llama", "local", 60.0, attempts_2, "post-window",
            config=config,
        )
        assert cooldown2 == 60.0, "Post-window 503 applies normal cooldown"

        # Reload → fresh window armed again.
        cold_start.note_model_loaded()
        assert cold_start.is_cold(config)

        attempts_3 = []
        cooldown3 = provider._handle_http_error_with_cooldown(
            response, "local-llama", "local", 60.0, attempts_3, "reloaded",
            config=config,
        )
        assert cooldown3 == 15, "Re-armed window: 503 is immune again"


# ── Failure counter not polluted during window ───────────────────────────────

class TestFailureCounterImmunity:
    """The exponential-backoff failure counter is not polluted during the window."""

    def test_failure_counter_unchanged_after_multiple_cold_503s(self):
        """Multiple cold-start 503s do not accumulate failure count."""
        config = _local_config()
        cold_start.note_model_loaded()

        response = MagicMock()
        response.status_code = 503
        response.headers = {}

        for _ in range(5):
            attempts = []
            provider._handle_http_error_with_cooldown(
                response, "local-llama", "local", 60.0, attempts, "cold",
                config=config,
            )

        assert provider._provider_failure_count.get("local-llama", 0) == 0, (
            "Failure counter must remain zero after multiple cold 503s"
        )

    def test_failure_counter_unchanged_after_multiple_cold_conn_errors(self):
        """Multiple cold connection errors do not accumulate failure count."""
        config = _local_config()
        cold_start.note_model_loaded()

        exc = httpx.ConnectError("Connection refused")
        for _ in range(5):
            attempts = []
            provider._handle_connection_error_in_fallback(
                exc, "local-llama", "local", 60.0, attempts, config=config,
            )

        assert provider._provider_failure_count.get("local-llama", 0) == 0, (
            "Failure counter must remain zero after multiple cold connection errors"
        )

    def test_failure_counter_resumes_after_window(self):
        """Quarantine resumes after the window; the counter stays un-polluted.

        The exponential-backoff counter is remote-only; the immunity guarantee
        is that deferrals during the window neither quarantine nor bump it, and
        quarantine resumes normally once the window has expired.
        """
        config = _local_config(local_cold_start_grace_seconds=10)
        cold_start.note_model_loaded()

        # Cold: deferral — no quarantine, no counter bump, deferral attempt.
        attempts_1 = []
        response = MagicMock()
        response.status_code = 503
        response.headers = {}
        provider._handle_http_error_with_cooldown(
            response, "local-llama", "local", 60.0, attempts_1, "cold",
            config=config,
        )
        assert not provider._is_provider_unavailable("local-llama")
        assert attempts_1[0]["status"] == "cold_start_deferral"

        # After window: quarantine resumes (real past for the handler clock).
        cold_start._state.loaded_at = time.monotonic() - 15
        attempts_2 = []
        provider._handle_http_error_with_cooldown(
            response, "local-llama", "local", 60.0, attempts_2, "post-window",
            config=config,
        )
        assert provider._is_provider_unavailable("local-llama"), (
            "Quarantine must resume once the window expires"
        )
        assert attempts_2[0]["status"] == "http_error"
        # Local never touches the failure counter (remote-only).
        assert provider._provider_failure_count.get("local-llama", 0) == 0


# ── Config absent / backwards compatibility ──────────────────────────────────

class TestBackwardsCompatibility:
    """When config is absent, existing test call sites work unchanged."""

    def test_handle_http_error_without_config(self):
        """_handle_http_error_with_cooldown works without config (existing callers)."""
        provider._provider_unavailable_until.clear()
        provider._provider_failure_count.clear()
        provider._usage_reset_at.clear()
        attempts = []
        response = MagicMock()
        response.status_code = 502
        response.headers = {}
        cooldown = provider._handle_http_error_with_cooldown(
            response, "test-provider", "remote", 60.0, attempts, "bad gateway",
        )
        assert cooldown == 1.0  # Remote exponential backoff
        assert provider._is_provider_unavailable("test-provider")

    def test_handle_connection_error_without_config(self):
        """_handle_connection_error_in_fallback works without config."""
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        result = provider._handle_connection_error_in_fallback(
            exc, "test-provider", "remote", 60.0, attempts,
        )
        assert result is True
        assert provider._is_provider_unavailable("test-provider")
