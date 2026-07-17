"""
Tests for the sliding-window stall circuit breaker (StallCircuitBreaker).

Tests cover:
1. AC1: Threshold triggering - N stalls within the window triggers cooldown
2. AC2: Window expiry - old stalls outside the window are ignored
3. AC3: Lazy pruning - expired timestamps pruned on each record_stall()
4. AC4: Cooldown enforcement - provider stays unavailable for cooldown_seconds
5. AC5: Config overrides - non-default parameters are respected
6. AC6: Thread-safety - concurrent access via asyncio.gather
7. AC7: No false trigger - fewer stalls than threshold does not trigger cooldown
8. AC8: Multiple providers - independent state per provider
"""

import asyncio
import time

import pytest

import proxy.provider as provider_mod


# ===================================================================
# Test helpers
# ===================================================================


class _TimeStepper:
    """Helper to control time.time() in tests via mock."""

    def __init__(self, start_time=1000.0):
        self._now = start_time

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __call__(self) -> float:
        return self._now


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset cooldown state between tests to avoid cross-test leakage."""
    provider_mod._provider_unavailable_until.clear()
    provider_mod._provider_failure_count.clear()
    yield


# ===================================================================
# AC7: No false trigger
# ===================================================================


def test_below_threshold_does_not_trigger(monkeypatch):
    """Fewer stalls than threshold within window does NOT trigger cooldown.

    AC7: With threshold=3, recording 2 stalls within the window should NOT
    mark the provider unavailable.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {"server": {"upstream_stall_window_seconds": 300, "upstream_stall_threshold": 3}}
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Record 2 stalls (below threshold of 3)
    cb.record_stall("provider-a")
    stepper.advance(10)
    cb.record_stall("provider-a")

    # Provider should NOT be in cooldown
    assert not provider_mod._is_provider_unavailable("provider-a"), (
        "Provider should not be marked unavailable with only 2 stalls (threshold=3)"
    )


# ===================================================================
# AC1: Threshold triggering
# ===================================================================


def test_threshold_triggered_marks_provider_unavailable(monkeypatch):
    """N stalls within the sliding window triggers cooldown.

    AC1: With threshold=3 and window=300s, recording 3 stalls within
    300s should call mark_provider_unavailable() and make the provider
    unavailable.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Record 3 stalls within the window
    triggered = False
    for i in range(3):
        triggered = cb.record_stall("provider-a")
        stepper.advance(5)  # 5 seconds between stalls

    # The 3rd stall should have triggered cooldown
    assert triggered, "record_stall() should return True when cooldown is triggered"
    assert provider_mod._is_provider_unavailable("provider-a"), (
        "Provider should be marked unavailable after threshold exceeded"
    )

    # Verify cooldown duration (~180s)
    expiry = provider_mod._provider_unavailable_until.get("provider-a")
    assert expiry is not None, "Provider should have a cooldown expiry timestamp"
    remaining = expiry - time.time()
    assert 170 <= remaining <= 190, (
        f"Cooldown should be ~180s, got ~{remaining:.0f}s remaining"
    )


# ===================================================================
# AC2: Window expiry
# ===================================================================


def test_stalls_outside_window_not_counted(monkeypatch):
    """Stalls older than the sliding window are ignored.

    AC2: With window=300s, stalls recorded 400s ago should not count
    toward the threshold. Recording 3 stalls with a 400s gap between
    each (within window for the total duration, but the oldest is too old)
    should not trigger.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Record 2 stalls close together, then wait >300s, then record 1 more
    cb.record_stall("provider-b")
    stepper.advance(10)
    cb.record_stall("provider-b")

    # Advance past the window (300s) so the first 2 stalls expire
    stepper.advance(350)

    # Only 1 stall remains in the window, but we add 2 more
    cb.record_stall("provider-b")  # 3rd stall within window
    stepper.advance(10)
    triggered = cb.record_stall("provider-b")  # 4th stall within window

    # Should have triggered on the 4th stall (3 within window: 3rd + 4th + one
    # of the earlier ones depends on timing — let's be more precise)
    # Actually: with 350s gap, the 1st stall (at t=0) is outside the 300s window
    # when the 3rd stall is recorded (at t=360). The 2nd stall (at t=10) expired
    # at t=310, also outside. So when recording stall 3 at t=360, only 1 stall
    # is in-window. After stall 4 at t=370, 2 stalls are in-window. Still below
    # threshold=3. Let me verify this more carefully...

    # Actually the assertion here is just that stale stalls are properly pruned.
    # Let me verify with a simpler approach:
    # Record 3 stalls with the oldest being stale (outside window)
    stepper2 = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper2)

    cb2 = StallCircuitBreaker(cb_cfg)
    cb2.record_stall("provider-c")  # t=1000
    stepper2.advance(350)           # t=1350 - this stall is now 350s old
    cb2.record_stall("provider-c")  # t=1350
    stepper2.advance(10)            # t=1360
    cb2.record_stall("provider-c")  # t=1360 - 3rd stall

    # At t=1360, the stall from t=1000 is 360s old (>300s window),
    # so only 2 stalls (t=1350, t=1360) are in the window.
    # threshold=3, so should NOT trigger
    assert not provider_mod._is_provider_unavailable("provider-c"), (
        "Provider should not be unavailable when old stalls expired from window"
    )


# ===================================================================
# AC3: Lazy pruning
# ===================================================================


def test_expired_timestamps_pruned_on_record(monkeypatch):
    """Expired timestamps are pruned lazily when a new stall is recorded.

    AC3: A provider with stale stall entries should have them pruned
    when record_stall() is called, without needing an explicit cleanup step.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Record 3 stalls, then wait >300s (all expire)
    cb.record_stall("provider-d")  # t=1000
    stepper.advance(10)
    cb.record_stall("provider-d")  # t=1010
    stepper.advance(10)
    cb.record_stall("provider-d")  # t=1020
    stepper.advance(350)           # t=1370 — all three stalls are >300s old

    # Internal state should still have 3 entries (lazy — not cleaned until
    # next record_stall)
    assert len(cb._stall_timestamps.get("provider-d", [])) == 3, (
        "Stale timestamps should still be present before lazy pruning"
    )

    # Recording a new stall triggers lazy pruning
    triggered = cb.record_stall("provider-d")  # t=1370

    # After pruning, only 1 stall (t=1370) should remain
    assert len(cb._stall_timestamps.get("provider-d", [])) == 1, (
        "Only the new stall should remain after lazy pruning"
    )
    # 1 stall < threshold=3, so should NOT trigger
    assert not triggered, "Should not trigger cooldown with only 1 stall in window"


# ===================================================================
# AC4: Cooldown enforcement
# ===================================================================


def test_provider_stays_unavailable_for_cooldown(monkeypatch):
    """Once triggered, provider stays unavailable for cooldown_seconds.

    AC4: After threshold exceeded and cooldown triggered, additional
    calls to record_stall() while in cooldown should be recorded but
    not extend the cooldown.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Record 3 stalls to trigger cooldown
    for i in range(3):
        cb.record_stall("provider-e")
        stepper.advance(5)

    assert provider_mod._is_provider_unavailable("provider-e")
    expiry_before = provider_mod._provider_unavailable_until.get("provider-e")

    # Record another stall while in cooldown
    stepper.advance(10)
    triggered = cb.record_stall("provider-e")

    # Provider should still be unavailable
    assert provider_mod._is_provider_unavailable("provider-e"), (
        "Provider should remain unavailable during cooldown"
    )

    # Cooldown should NOT have been extended (original expiry is ~seconds away)
    expiry_after = provider_mod._provider_unavailable_until.get("provider-e")
    assert expiry_after == expiry_before, (
        "Cooldown should not be extended by stalls during cooldown"
    )

    # record_stall returns False when cooldown is NOT newly triggered
    # (already in cooldown)
    assert not triggered, (
        "record_stall() should return False when already in cooldown"
    )


# ===================================================================
# AC5: Config overrides
# ===================================================================


def test_config_overrides(monkeypatch):
    """Non-default config parameters are respected.

    AC5: Custom window, threshold, and cooldown values are used
    instead of defaults.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    # Custom config: very short window (10s), low threshold (2), short cooldown (30s)
    config = {
        "server": {
            "upstream_stall_window_seconds": 10,
            "upstream_stall_threshold": 2,
            "upstream_stall_cooldown_seconds": 30,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # 2 stalls within 10s should trigger
    cb.record_stall("provider-f")
    stepper.advance(1)
    triggered = cb.record_stall("provider-f")

    assert triggered, (
        "Should trigger cooldown with threshold=2 and 2 stalls in window=10s"
    )

    # Verify custom cooldown (~30s, not 180s default)
    expiry = provider_mod._provider_unavailable_until.get("provider-f")
    assert expiry is not None
    remaining = expiry - time.time()
    assert 25 <= remaining <= 35, (
        f"Cooldown should be ~30s (custom), got ~{remaining:.0f}s remaining"
    )


# ===================================================================
# AC5b: Config defaults used when keys absent
# ===================================================================


def test_config_defaults(monkeypatch):
    """Default config values apply when config keys are absent.

    AC5: When config has no circuit breaker keys, the default values
    (window=300, threshold=3, cooldown=180) are used.
    """
    from proxy.stall_circuit_breaker import _get_circuit_breaker_config

    # Config with no circuit breaker keys
    config_empty = {}
    cfg_empty = _get_circuit_breaker_config(config_empty)
    assert cfg_empty["window_seconds"] == 300, (
        f"Default window should be 300, got {cfg_empty['window_seconds']}"
    )
    assert cfg_empty["threshold"] == 3, (
        f"Default threshold should be 3, got {cfg_empty['threshold']}"
    )
    assert cfg_empty["cooldown_seconds"] == 180, (
        f"Default cooldown should be 180, got {cfg_empty['cooldown_seconds']}"
    )

    # Test with partial server config (no circuit breaker keys)
    config_partial = {"server": {}}
    cfg_partial = _get_circuit_breaker_config(config_partial)
    assert cfg_partial["window_seconds"] == 300
    assert cfg_partial["threshold"] == 3
    assert cfg_partial["cooldown_seconds"] == 180

    # Test with nested server config
    config_full = {
        "server": {
            "upstream_stall_window_seconds": 600,
            "upstream_stall_threshold": 5,
            "upstream_stall_cooldown_seconds": 300,
        }
    }
    cfg_full = _get_circuit_breaker_config(config_full)
    assert cfg_full["window_seconds"] == 600
    assert cfg_full["threshold"] == 5
    assert cfg_full["cooldown_seconds"] == 300


# ===================================================================
# AC6: Thread-safety (asyncio-safe concurrent access)
# ===================================================================


@pytest.mark.asyncio
async def test_concurrent_stall_recording(monkeypatch):
    """Concurrent stall recordings don't corrupt internal state.

    AC6: Multiple coroutines recording stalls for the same provider
    simultaneously should not lose any events or corrupt the deque.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 10,  # high threshold to avoid triggering
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    async def record_concurrently(provider: str, count: int):
        tasks = []
        for i in range(count):
            tasks.append(asyncio.to_thread(cb.record_stall, provider))
            stepper.advance(0.001)  # slight time advancement for unique timestamps
        await asyncio.gather(*tasks)

    # Record 20 stalls concurrently for the same provider
    await record_concurrently("provider-g", 20)

    # All 20 should be in the deque (or pruned if window expired,
    # but with 0.001s * 20 = 0.02s, none expired yet in 300s window)
    assert len(cb._stall_timestamps.get("provider-g", [])) == 20, (
        f"Expected 20 timestamps, got {len(cb._stall_timestamps.get('provider-g', []))}"
    )


# ===================================================================
# AC8: Multiple providers have independent state
# ===================================================================


def test_multiple_providers_independent(monkeypatch):
    """Stalls for one provider do not affect another.

    AC8: Recording stalls for provider-a should not affect the
    circuit breaker state for provider-b.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # Saturate provider-a with 3 stalls (triggers cooldown)
    for i in range(3):
        cb.record_stall("provider-a")
        stepper.advance(5)

    assert provider_mod._is_provider_unavailable("provider-a"), (
        "provider-a should be in cooldown"
    )

    # provider-b should NOT be in cooldown
    assert not provider_mod._is_provider_unavailable("provider-b"), (
        "provider-b should NOT be in cooldown (independent state)"
    )

    # provider-b should still allow normal stall recording
    cb.record_stall("provider-b")
    assert not provider_mod._is_provider_unavailable("provider-b"), (
        "provider-b with 1 stall should not trigger"
    )


# ===================================================================
# Edge case: Clean provider (no prior stalls)
# ===================================================================


def test_clean_provider_no_stalls(monkeypatch):
    """A provider with no recorded stalls should not be affected.

    Verifies that accessing a never-seen-before provider works correctly.
    """
    from proxy.stall_circuit_breaker import StallCircuitBreaker, _get_circuit_breaker_config

    config = {}
    cb_cfg = _get_circuit_breaker_config(config)
    cb = StallCircuitBreaker(cb_cfg)

    # No stalls recorded for unknown-provider
    assert cb._stall_timestamps.get("unknown-provider", []) == [], (
        "Unknown provider should have empty stall list"
    )
    assert not provider_mod._is_provider_unavailable("unknown-provider"), (
        "Unknown provider should not be unavailable"
    )


# ===================================================================
# Edge case: Singleton instance import
# ===================================================================


def test_stall_circuit_breaker_is_singleton(monkeypatch):
    """The module-level stall_circuit_breaker instance is a singleton.

    Verifies that importing stall_circuit_breaker returns the same
    instance on repeated access.
    """
    # First import
    from proxy.stall_circuit_breaker import stall_circuit_breaker as cb1

    # Re-import should give same instance
    import proxy.stall_circuit_breaker as scb_mod
    cb2 = scb_mod.stall_circuit_breaker

    assert cb1 is cb2, (
        "stall_circuit_breaker should be a module-level singleton"
    )


# ===================================================================
# Edge case: _check_stall_circuit_breaker function integration
# ===================================================================


def test_check_stall_circuit_breaker_integration(monkeypatch):
    """The _check_stall_circuit_breaker() function records a stall and
    triggers cooldown if threshold exceeded.

    This is the integration point called from proxy_remote.py.
    """
    from proxy.stall_circuit_breaker import _check_stall_circuit_breaker

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 60,
        }
    }

    # Record 3 stalls
    for i in range(3):
        _check_stall_circuit_breaker("provider-h", config)
        stepper.advance(5)

    # Provider should be in cooldown
    assert provider_mod._is_provider_unavailable("provider-h"), (
        "Provider should be in cooldown after _check_stall_circuit_breaker threshold"
    )


def test_check_stall_circuit_breaker_below_threshold(monkeypatch):
    """The _check_stall_circuit_breaker() function does not trigger when
    below threshold.
    """
    from proxy.stall_circuit_breaker import _check_stall_circuit_breaker

    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 5,  # high threshold
        }
    }

    # Record 2 stalls (below threshold)
    _check_stall_circuit_breaker("provider-i", config)
    stepper.advance(10)
    _check_stall_circuit_breaker("provider-i", config)

    # Provider should NOT be in cooldown
    assert not provider_mod._is_provider_unavailable("provider-i"), (
        "Provider should NOT be in cooldown when below threshold"
    )


# ===================================================================
# Edge case: Backward compatibility - no server config at all
# ===================================================================


def test_no_config_dot_server(monkeypatch):
    """When config has no 'server' key, defaults still apply."""
    from proxy.stall_circuit_breaker import _get_circuit_breaker_config

    cfg = _get_circuit_breaker_config({})
    assert cfg["window_seconds"] == 300
    assert cfg["threshold"] == 3
    assert cfg["cooldown_seconds"] == 180
