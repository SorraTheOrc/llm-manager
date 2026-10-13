"""AC2 re-scope metric helpers: warm-cache classification + queue-wait split.

Covers the LP-0MV6CFMU9005WU2E instrumentation for the re-scoped AC2 of
LP-0MU1S617L002Q9PR: the ``dispatch_first_byte_ms`` log line for local
dispatches is segmented so warm-cache dispatches are identifiable
(``cache_state=warm|cold`` from the routing ``cached_ratio``) and the
single-slot queue wait is separated (``queue_wait_ms``) from service time
(``dispatch_to_first_byte_ms``).

Warm/cold classification must be fail-open: an absent ratio, a missing
( model, session ) cache entry, or a non-numeric value all classify COLD so
the conservative target (warm-cache p99) is never accidentally diluted by
unknown cases.
"""

from __future__ import annotations

import proxy.provider as provider
import pytest
from proxy.router_helpers import (
    _WARM_CACHE_RATIO_THRESHOLD,
    _classify_cache_state,
    _lookup_dispatch_cache_state,
)


@pytest.fixture(autouse=True)
def reset_cached_ratio():
    provider._last_cached_ratio.clear()
    yield


# ===================================================================
# AC1 — warm/cold classification is deterministic and fail-open
# ===================================================================


def test_warm_threshold_is_realistic_and_documented():
    assert _WARM_CACHE_RATIO_THRESHOLD == 0.5


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (1.0, "warm"),
        (0.9, "warm"),
        (0.5, "warm"),
        (0.499999, "cold"),
        (0.1, "cold"),
        (0.0, "cold"),
    ],
)
def test_classify_cache_state_threshold(ratio, expected):
    assert _classify_cache_state(ratio) == expected


def test_classify_cache_state_fail_open():
    """Non-numeric / missing values classify COLD (never warm)."""
    assert _classify_cache_state(None) == "cold"
    assert _classify_cache_state("n/a") == "cold"
    assert _classify_cache_state(float("nan")) == "cold"


# ===================================================================
# AC2 — lookup surfaces the routing cached_ratio per (model, session)
# ===================================================================


def test_lookup_warm_uses_provider_ratio():
    provider._last_cached_ratio[("Qwen3", "sess-1")] = 0.93
    state, ratio = _lookup_dispatch_cache_state("Qwen3", "sess-1")
    assert state == "warm"
    assert ratio == pytest.approx(0.93)


def test_lookup_cold_when_ratio_absent():
    state, ratio = _lookup_dispatch_cache_state("Qwen3", "sess-absent")
    assert state == "cold"
    assert ratio == pytest.approx(0.0)


def test_lookup_cold_when_session_unknown():
    """No session id (or provider rejecting a falsy session) => cold, never
    an exception in the logging path."""
    state, ratio = _lookup_dispatch_cache_state("Qwen3", None)
    assert state == "cold"
    assert ratio == pytest.approx(0.0)

    state, ratio = _lookup_dispatch_cache_state("Qwen3", "unknown")
    assert state == "cold"
    assert ratio == pytest.approx(0.0)


def test_lookup_fail_open_on_provider_error(monkeypatch):
    """Even if the provider lookup explodes, the metric path still resolves
    to cold/0.0 instead of raising into the streaming loop."""
    def _boom(_m, _s):
        raise RuntimeError("boom")

    monkeypatch.setattr(provider, "_get_cached_ratio", _boom)
    state, ratio = _lookup_dispatch_cache_state("Qwen3", "sess-2")
    assert state == "cold"
    assert ratio == pytest.approx(0.0)
