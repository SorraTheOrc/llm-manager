"""
Unit tests for the sliding-window rate limiter (proxy/rate_limiter.py).

Tests cover:
- Basic accept/reject behaviour at the boundary
- Sliding window expiry (old entries don't count)
- Unlimited mode (max_rpm=0)
- Reset
- Integration into provider fallback: rate-limited providers are skipped
- remaining() helper
"""

import time
from unittest.mock import AsyncMock, patch

import pytest
from proxy.rate_limiter import SlidingWindowRateLimiter, get_rate_limiter

# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def limiter():
    """Return a fresh limiter for each test."""
    return SlidingWindowRateLimiter()


@pytest.mark.asyncio
async def test_allows_requests_under_limit(limiter):
    """Requests within the limit should be allowed."""
    for i in range(5):
        allowed = await limiter.check_and_increment("test-provider", max_rpm=10, window_seconds=60)
        assert allowed, f"Request {i+1} should be allowed (under 10 rpm)"


@pytest.mark.asyncio
async def test_rejects_requests_over_limit(limiter):
    """Requests exceeding the limit should be rejected."""
    for i in range(5):
        allowed = await limiter.check_and_increment("test-provider", max_rpm=5, window_seconds=60)
        assert allowed, f"Request {i+1} should be allowed"

    # 6th request should be rejected
    allowed = await limiter.check_and_increment("test-provider", max_rpm=5, window_seconds=60)
    assert not allowed, "6th request should be rejected (limit is 5 rpm)"


@pytest.mark.asyncio
async def test_unlimited_when_max_rpm_is_zero(limiter):
    """max_rpm=0 means unlimited — all requests should be allowed."""
    for i in range(100):
        allowed = await limiter.check_and_increment("unlimited", max_rpm=0, window_seconds=60)
        assert allowed, f"Request {i+1} should be allowed when max_rpm=0"


@pytest.mark.asyncio
async def test_sliding_window_expiry(limiter):
    """Old entries should expire after the window passes."""
    # Freeze time for the test
    original_monotonic = time.monotonic

    try:
        fake_now = 1000.0
        _call_count = 0

        def _fake_monotonic():
            nonlocal fake_now
            return fake_now

        time.monotonic = _fake_monotonic  # type: ignore[assignment]

        # Send 3 requests at t=1000 (limit is 3 per 60s)
        for _ in range(3):
            allowed = await limiter.check_and_increment("sliding-provider", max_rpm=3, window_seconds=60)
            assert allowed

        # 4th request at same time — should be rejected
        allowed = await limiter.check_and_increment("sliding-provider", max_rpm=3, window_seconds=60)
        assert not allowed, "4th request should be rejected"

        # Advance past the window (t=1061, > 60s after first request)
        fake_now = 1061.0
        allowed = await limiter.check_and_increment("sliding-provider", max_rpm=3, window_seconds=60)
        assert allowed, "Request after window expiry should be allowed"
    finally:
        time.monotonic = original_monotonic


@pytest.mark.asyncio
async def test_independent_windows_per_key(limiter):
    """Different keys should have independent counters."""
    for i in range(5):
        allowed_a = await limiter.check_and_increment("provider-a", max_rpm=5, window_seconds=60)
        allowed_b = await limiter.check_and_increment("provider-b", max_rpm=5, window_seconds=60)
        assert allowed_a and allowed_b, "Both providers should allow up to 5 requests"

    # provider-a should be full now
    assert not await limiter.check_and_increment("provider-a", max_rpm=5, window_seconds=60)
    # provider-b should still be under limit (5th was just sent above, but that's exactly 5)
    # Actually we sent 5 to provider-b, so the 6th should fail too
    assert not await limiter.check_and_increment("provider-b", max_rpm=5, window_seconds=60)


@pytest.mark.asyncio
async def test_reset_clears_counter(limiter):
    """reset() should clear all tracked data for the key."""
    for _ in range(5):
        await limiter.check_and_increment("reset-me", max_rpm=5, window_seconds=60)

    # Should be at limit
    assert not await limiter.check_and_increment("reset-me", max_rpm=5, window_seconds=60)

    # Reset
    limiter.reset("reset-me")
    allowed = await limiter.check_and_increment("reset-me", max_rpm=5, window_seconds=60)
    assert allowed, "Request should be allowed after reset"


def test_remaining_returns_capacity(limiter):
    """remaining() should return the number of available slots."""
    # No requests yet — should show full capacity
    assert limiter.remaining("capacity-test", max_rpm=10, window_seconds=60) == 10

    # Add some requests manually (sync access for test)
    limiter._windows["capacity-test"].extend([time.monotonic()] * 3)

    remaining = limiter.remaining("capacity-test", max_rpm=10, window_seconds=60)
    assert remaining == 7, f"Expected 7 remaining, got {remaining}"


def test_remaining_unlimited_returns_zero(limiter):
    """remaining() should return 0 for unlimited providers."""
    assert limiter.remaining("unlimited", max_rpm=0, window_seconds=60) == 0


def test_get_rate_limiter_singleton():
    """get_rate_limiter() should return the same instance each time."""
    instance1 = get_rate_limiter()
    instance2 = get_rate_limiter()
    assert instance1 is instance2


# ---------------------------------------------------------------------------
# Integration test: rate limiter in provider fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_check_called_in_provider_fallback():
    """The rate limiter check is called for remote providers that have
    ``rate_limit_rpm`` configured.  This verifies the integration point
    by patching the rate limiter and observing that ``check_and_increment``
    is called when the provider config includes ``rate_limit_rpm``.
    """
    from proxy.provider import proxy_with_remote_fallback
    from proxy.rate_limiter import get_rate_limiter

    get_rate_limiter().reset("test-rl-provider")
    get_rate_limiter().reset("test-fallback")

    config = {
        "server": {
            "provider_cooldown_seconds": 0,
        }
    }

    model_config = {
        "providers": [
            {
                "name": "test-rl-provider",
                "type": "remote",
                "endpoint": "https://fake.example.com/v1",
                "rate_limit_rpm": 5,
            },
        ]
    }

    # Patch the remote proxy function so we don't hit the network
    # NOTE (LP-0MVDUN5CY009VD20): with ``provider_rate_limit_reserve_ratio``
    # unset, the reserve defaults to 0.3, so a session with no affinity (these
    # requests carry no session header) may use only ``rpm - reserve`` = 4 of
    # the 5 slots; the 5th request is skipped and the chain is exhausted.
    with patch("proxy.provider._get_proxy_to_remote") as mock_get_ptr:
        mock_ptr = AsyncMock()
        mock_get_ptr.return_value = mock_ptr

        # Make the remote call return a simple 429 so the test flow is simple
        mock_ptr.return_value = type("Resp", (), {
            "status_code": 429,
            "headers": type("H", (), {"get": lambda self, k: None, "append": lambda self, k, v: None})(),
            "body": b'{}',
            "content": b'{}',
            "__class__": type("C", (), {"__name__": "Response"}),
        })()

        request = type("Req", (), {
            "method": "POST",
            "url": type("U", (), {"path": "/v1/chat/completions"})(),
            "headers": {},
        })()

        # Call the fallback function.  The rate limiter should allow the first
        # few requests and return the upstream 429.  Since there's only one
        # provider, the result should be an "all providers exhausted" response.
        _resp = await proxy_with_remote_fallback(request, "v1/chat/completions", model_config, config)

        # The rate limiter should have been checked and incremented for
        # "test-rl-provider". The reserve (default 3/10 → floor(5*0.3)=1 slot
        # here) keeps a slot free for in-progress sessions, so an anonymous
        # request may only use 4 of the 5 slots.
        remaining = get_rate_limiter().remaining("test-rl-provider", max_rpm=5)
        assert remaining == 4, (
            f"Expected 4 remaining requests after first call, got {remaining}"
        )

        # Now exhaust the unreserved part of the window with 3 more calls
        for i in range(3):
            await proxy_with_remote_fallback(request, "v1/chat/completions", model_config, config)

        remaining = get_rate_limiter().remaining("test-rl-provider", max_rpm=5)
        assert remaining == 1, (
            f"Expected the reserved slot to remain free, got {remaining}"
        )

        # A 5th anonymous call is skipped (the reserved slot is not for it)
        await proxy_with_remote_fallback(request, "v1/chat/completions", model_config, config)

        # The rate limiter tracked 4 requests: check_and_increment was called
        # before each remote dispatch; the reserved slot was never used.
        assert mock_ptr.call_count == 4, (
            f"Expected 4 remote calls (one per request before the reserve blocked), "
            f"got {mock_ptr.call_count}"
        )

    get_rate_limiter().reset("test-rl-provider")
    get_rate_limiter().reset("test-fallback")
