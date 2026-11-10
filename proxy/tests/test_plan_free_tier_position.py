"""Free-tier positioning in the ``plan`` chain (LP-0MVWG4U7F0009LTH).

The ``plan`` chain used to place ``munitoai-free`` immediately after the local
Qwen3 tier, so a request that overflowed local consumed free tokens *before*
the timed ``opencode-go`` accounts — even inside their ``available_times``
windows. Free tokens are a daily-reset allowance (LP-0MVT8SAZE00570J5), so
spending them while a costed tier is eligible loses capacity that is only
useful during the gap periods (UTC 01:00-04:00 and 06:00-10:00,
LP-0MU1RXEPL005CK97), when the opencode accounts are unavailable.

``munitoai-free`` therefore sits after all three ``opencode-go`` entries and
before the paid ``munitoai`` tier (``author``/``code`` keep the original
order). This module asserts the **routing consequence** hour by hour rather
than restating the list: the shipped chain order itself is frozen by
``test_provider_priorities_config.py`` and
``test_low_priority_offpeak_fallback.py``, while the assertions here run
against the live ``proxy/config.yaml`` entries through the seams the router
uses (``provider_chain_for_priority`` and
``provider._resolve_provider_with_exclusions``), so they cannot drift from the
shipped chain and they observe the real window gate instead of a copy of it.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import proxy.provider as provider
import pytest
import yaml
from proxy.priority import PRIORITY_LOW, provider_chain_for_priority, proxy_dir

PLAN = "plan"
LOCAL_TIER = "local-qwen3"
FREE_TIER = "munitoai-free"
OPENCODE_NAMES = ["opencode-go-3", "opencode-go-2", "opencode-go"]

#: Hours inside an opencode ``available_times`` window — one from each of the
#: three shipped windows (``00:00-01:00``, ``04:00-06:00``, ``10:00-00:00``).
#: A costed account is eligible at every one of them.
INSIDE_OPENCODE_WINDOW_UTC = [
    datetime(2026, 1, 5, 0, 30, tzinfo=UTC),
    datetime(2026, 1, 5, 4, 30, tzinfo=UTC),
    datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
]

#: Hours inside the opencode gap periods (``01:00-04:00``, ``06:00-10:00``),
#: where the free tier is the first non-local tier that can serve.
GAP_UTC = [
    datetime(2026, 1, 5, 2, 0, tzinfo=UTC),
    datetime(2026, 1, 5, 8, 0, tzinfo=UTC),
]


class _FakeDatetime:
    """Deterministic UTC clock for the window gate inside the resolver.

    Mirrors the convention already used by
    ``test_low_priority_offpeak_fallback.py`` so the resolver observes a fixed
    ``now`` without touching the production seam.
    """

    _now = INSIDE_OPENCODE_WINDOW_UTC[0]

    @classmethod
    def now(cls, tz=None):
        return cls._now

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)


def _patch_utc_now(moment):
    _FakeDatetime._now = moment
    return patch("proxy.provider.datetime", _FakeDatetime)


@pytest.fixture(autouse=True)
def reset_provider_state():
    """Clear cross-request quarantine state before and after each test."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()


def _live_plan_model() -> dict:
    with open(proxy_dir() / "config.yaml") as fh:
        return yaml.safe_load(fh)["models"][PLAN]


def _live_entry(name: str) -> dict:
    for entry in _live_plan_model()["providers"]:
        if entry.get("name") == name:
            return entry
    raise AssertionError(f"{name!r} not found in the {PLAN!r} chain")


def _next_provider(chain: list[dict], named: set[str]) -> dict | None:
    return provider._resolve_provider_with_exclusions({"providers": chain}, set(named), set())


@pytest.mark.parametrize("moment", INSIDE_OPENCODE_WINDOW_UTC, ids=lambda m: m.strftime("%H:%M"))
def test_in_window_requests_spend_a_costed_account_before_free_tokens(moment):
    """AC2: inside an opencode window, local overflow lands on opencode-go-3.

    The first eligible costed account absorbs the request, so the free
    allowance is not spent while a costed tier is available.
    """
    chain = provider_chain_for_priority(_live_plan_model(), PRIORITY_LOW)
    with _patch_utc_now(moment):
        picked = _next_provider(chain, {LOCAL_TIER})
    assert picked is not None
    assert picked["name"] == OPENCODE_NAMES[0]


@pytest.mark.parametrize("moment", GAP_UTC, ids=lambda m: m.strftime("%H:%M"))
def test_gap_period_requests_fall_through_to_the_free_tier(moment):
    """AC3: in an opencode gap the free tier serves the local overflow.

    Every opencode entry is outside its window there, so the free tier is the
    first non-local tier that can serve — free tokens are spent exactly where
    they are the only free capacity left.
    """
    chain = provider_chain_for_priority(_live_plan_model(), PRIORITY_LOW)
    with _patch_utc_now(moment):
        picked = _next_provider(chain, {LOCAL_TIER})
    assert picked is not None
    assert picked["name"] == FREE_TIER


@pytest.mark.parametrize("moment", INSIDE_OPENCODE_WINDOW_UTC, ids=lambda m: m.strftime("%H:%M"))
def test_free_tier_still_serves_when_the_opencode_accounts_are_exhausted(moment):
    """Inside a window the free tier remains the last free resort.

    Repositioning delays the free tier; it does not remove it. Once all three
    opencode accounts are exhausted the free tier is picked before any paid
    tier, so free capacity is still preferred over spend.
    """
    chain = provider_chain_for_priority(_live_plan_model(), PRIORITY_LOW)
    # Usage-limit account keys travel in the ``excluded_domains`` set, which is
    # the set the resolver checks them against.
    exhausted_accounts = {
        provider._usage_limit_account_key(_live_entry(name)) for name in OPENCODE_NAMES
    }
    with _patch_utc_now(moment):
        picked = provider._resolve_provider_with_exclusions(
            {"providers": chain}, {LOCAL_TIER}, exhausted_accounts
        )
    assert picked is not None
    assert picked["name"] == FREE_TIER


def test_only_the_position_of_the_free_entry_changed():
    """The repositioning must not silently re-tune the free tier (constraints).

    The budget, the reset-header handling, the tier-distinct display name, the
    context ceiling and the round-the-clock availability all stay exactly as
    shipped — only the chain position moved.
    """
    entry = _live_entry(FREE_TIER)
    assert entry["display_name"] == "MunitoAI Free"
    assert entry["api_key_env"] == "MUNITOAI_FREE_API_KEY"
    assert entry["rate_limit_rpm"] == 3
    assert entry["respect_ratelimit_headers"] is True
    assert entry["max_context_tokens"] == 150_000
    # No window: the free tier must stay eligible during the opencode gaps.
    assert "available_times" not in entry
