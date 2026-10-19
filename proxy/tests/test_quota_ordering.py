"""Tests for headroom-aware opencode chain ordering (LP-0MVF7KIKE002DZVD).

Every test seeds the quota cache through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) and then
asserts the routing decision through the public resolution API — no test
reaches into provider internals beyond the three resolution paths the
acceptance criteria name (``resolve_provider``,
``_resolve_provider_with_exclusions``, ``_has_next_provider``).

Covered acceptance criteria:

- AC3: the account with more effective headroom is preferred; reversing the
  percentages reverses the preference; equal headroom falls back to config
  order deterministically.
- AC3b: the ranking applies to first-attempt resolution and to mid-chain
  fallback (including ``_has_next_provider``), while non-``opencode-*``
  entries keep their configured positions.
- AC3c: the lowest-ranked entry is still tried when every higher-ranked
  sibling is excluded — ranking never drops candidates.
- AC3d: ordering decisions are logged at DEBUG with provider / headroom /
  seconds_to_reset / score / rank; quota-driven skips are logged at INFO
  with the account and reason, and ordering never logs at INFO.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import proxy.quota_state as quota_state
import pytest
from proxy.provider import (
    _has_next_provider,
    _quota_ordering_active,
    _quota_ranked_providers,
    _resolve_provider_with_exclusions,
    resolve_provider,
)
from proxy.quota_state import reset_quota_state

NOW = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"


# ---------------------------------------------------------------------------
# Fixtures / seeding helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def quota_cache():
    """Fresh quota cache and a frozen clock for every test."""
    reset_quota_state()
    quota_state.set_clock(lambda: NOW)
    yield
    reset_quota_state()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_body(
    *,
    monthly_percent: float,
    rolling_percent: float = 0.0,
    weekly_percent: float = 0.0,
    monthly_reset_days: float = 3.0,
    monthly_status: str = "ok",
) -> dict:
    """A ``/v1/usage`` body with a configurable monthly consumption.

    Rolling/weekly consumption defaults to ``0`` so the effective headroom is
    governed by the monthly window, and the monthly reset sits inside the
    default 7-day surge window so ``urgency_score`` has an opinion.
    """
    return {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": rolling_percent,
                "resetsAt": _iso(NOW + timedelta(hours=5)),
            },
            "weekly": {
                "status": "ok",
                "percent": weekly_percent,
                "resetsAt": _iso(NOW + timedelta(days=2)),
            },
            "monthly": {
                "status": monthly_status,
                "percent": monthly_percent,
                "resetsAt": _iso(NOW + timedelta(days=monthly_reset_days)),
            },
        }
    }


def _provider(name: str, api_key_env: str, model: str = "deepseek-v4-flash", *, brand: str = "opencode-go") -> dict:
    return {
        "name": name,
        "type": "remote",
        "provider": brand,
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": api_key_env,
        "model": model,
    }


def _non_quota_provider(name: str, api_key_env: str) -> dict:
    """A remote entry that is NOT quota-managed (its brand is not opencode)."""
    return {
        "name": name,
        "type": "remote",
        "provider": name,
        "endpoint": f"https://{name}.example.com/v1",
        "api_key_env": api_key_env,
        "model": "deepseek-v4-flash",
    }


def _config(**server: object) -> dict:
    return {"server": dict(server)}


class _Transport:
    """Injectable ``/v1/usage`` transport keyed by account."""

    def __init__(self, bodies: dict[str, dict]):
        self.bodies = dict(bodies)
        self.calls: list[str] = []

    async def __call__(self, provider_cfg: dict) -> dict:
        key = quota_state.account_key(provider_cfg)
        self.calls.append(key)
        return self.bodies[key]


async def _seed(model_config: dict, config: dict, bodies: dict[str, dict]) -> None:
    """Populate the quota cache through the public poller seam."""
    quota_state.set_quota_transport(_Transport(bodies))
    await quota_state.refresh_quota_accounts(model_config, config, force=True)


def _names(providers: list[dict] | None) -> list[str]:
    return [str(p.get("name")) for p in providers or []]


def _two_account_chain() -> dict:
    return {
        "providers": [
            _provider("opencode-go", "OPENCODE_KEY_A"),
            _provider("opencode-go-3", "OPENCODE_KEY_B"),
        ]
    }


# ---------------------------------------------------------------------------
# AC3 — headroom drives the preference, config order breaks ties
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefers_account_with_more_headroom():
    """88% consumed vs 5% consumed: the fuller account is tried first."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    assert resolve_provider(model_config, config=config)["name"] == "opencode-go-3"
    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go-3",
        "opencode-go",
    ]


@pytest.mark.asyncio
async def test_preference_reverses_with_percentages():
    """Reversing the headroom ordering reverses the routing preference."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=5),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=88),
        },
    )

    assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go",
        "opencode-go-3",
    ]


@pytest.mark.asyncio
async def test_equal_headroom_preserves_config_order():
    """Equal scores tie-break on config order, identically on every call."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=50),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=50),
        },
    )

    first = _names(_quota_ranked_providers(model_config, config))
    second = _names(_quota_ranked_providers(model_config, config))
    assert first == second == ["opencode-go", "opencode-go-3"]
    assert resolve_provider(model_config, config=config)["name"] == "opencode-go"


@pytest.mark.asyncio
async def test_no_snapshot_preserves_config_order():
    """Never-polled accounts score 0: the chain is exactly today's chain."""
    model_config = _two_account_chain()

    assert _names(_quota_ranked_providers(model_config, _config())) == [
        "opencode-go",
        "opencode-go-3",
    ]
    # No snapshot means no quota opinion: nothing is dropped either.
    assert resolve_provider(model_config, config=_config())["name"] == "opencode-go"


# ---------------------------------------------------------------------------
# AC3b — every resolution path ranks; other providers keep their positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_opencode_entries_keep_configured_positions():
    """Only quota entries are re-ordered; siblings stay at their index."""
    model_config = {
        "providers": [
            _provider("opencode-go", "OPENCODE_KEY_A"),
            _non_quota_provider("deepseek", "DEEPSEEK_KEY"),
            _provider("opencode-go-3", "OPENCODE_KEY_B"),
            _non_quota_provider("munitoai", "MUNITOAI_KEY"),
        ]
    }
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go-3",
        "deepseek",
        "opencode-go",
        "munitoai",
    ]


@pytest.mark.asyncio
async def test_mid_chain_fallback_uses_same_ranking():
    """Mid-chain resolution walks the ranked chain, not the config chain."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    first = _resolve_provider_with_exclusions(model_config, set(), set(), config=config)
    assert first["name"] == "opencode-go-3"

    second = _resolve_provider_with_exclusions(
        model_config, {"opencode-go-3"}, set(), config=config
    )
    assert second["name"] == "opencode-go"


@pytest.mark.asyncio
async def test_has_next_provider_agrees_with_resolution():
    """``_has_next_provider`` sees the same ranked chain as resolution."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    assert _has_next_provider(model_config, set(), set(), config=config) is True
    assert _has_next_provider(model_config, {"opencode-go-3"}, set(), config=config) is True
    assert (
        _has_next_provider(
            model_config, {"opencode-go", "opencode-go-3"}, set(), config=config
        )
        is False
    )


# ---------------------------------------------------------------------------
# AC3c — ranking degrades to the existing chain, never drops candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lowest_ranked_entry_still_attempted():
    """Every higher-ranked sibling excluded: the last-ranked entry is used."""
    model_config = {
        "providers": [
            _provider("opencode-go", "OPENCODE_KEY_A"),
            _provider("opencode-go-2", "OPENCODE_KEY_B"),
            _provider("opencode-go-3", "OPENCODE_KEY_C"),
        ]
    }
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=95),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=90),
            "OPENCODE_KEY_C": _usage_body(monthly_percent=85),
        },
    )

    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go-3",
        "opencode-go-2",
        "opencode-go",
    ]
    survivor = _resolve_provider_with_exclusions(
        model_config, {"opencode-go-3", "opencode-go-2"}, set(), config=config
    )
    assert survivor["name"] == "opencode-go"
    assert (
        _has_next_provider(
            model_config, {"opencode-go-3", "opencode-go-2"}, set(), config=config
        )
        is True
    )


# ---------------------------------------------------------------------------
# Exhausted accounts are unavailable (goal) — fail open without a snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_entry_is_skipped_with_info_log(caplog):
    """A spent account reached by the chain is skipped at INFO with the reason.

    The plain resolution prefers the healthy sibling, so the exhausted entry
    is only *iterated* once the healthy one is excluded — that is when the
    INFO skip fires.
    """
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=100),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        chosen = resolve_provider(model_config, config=config)
        assert chosen["name"] == "opencode-go-3"
        # The healthy sibling is excluded: the exhausted entry is now reached
        # and must be skipped rather than dispatched.
        survivor = _resolve_provider_with_exclusions(
            model_config, {"opencode-go-3"}, set(), config=config
        )
    assert survivor is None
    skips = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "quota_exhausted" in record.getMessage()
    ]
    assert skips, "an exhausted quota account must be skipped at INFO"
    assert "opencode-go" in skips[0]
    assert "OPENCODE_KEY_A" in skips[0]


@pytest.mark.asyncio
async def test_all_quota_entries_exhausted_returns_none():
    """When every quota entry is spent the chain reports no provider."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=100),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=100),
        },
    )

    assert resolve_provider(model_config, config=config) is None
    assert (
        _resolve_provider_with_exclusions(model_config, set(), set(), config=config)
        is None
    )
    assert _has_next_provider(model_config, set(), set(), config=config) is False


# ---------------------------------------------------------------------------
# AC3d — DEBUG ordering decisions, INFO-only skips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordering_decisions_logged_at_debug_not_info(caplog):
    """Each ranked entry is logged at DEBUG with the documented fields."""
    model_config = _two_account_chain()
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    with caplog.at_level(logging.DEBUG, logger=_PROVIDER_LOGGER):
        resolve_provider(model_config, config=config)

    debug_lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and "quota_ordering" in record.getMessage()
    ]
    assert len(debug_lines) == 2
    assert "rank=1" in debug_lines[0]
    assert "provider=opencode-go-3" in debug_lines[0]
    for field in ("headroom=", "seconds_to_reset=", "score="):
        assert field in debug_lines[0]
    assert "rank=2" in debug_lines[1]
    assert "provider=opencode-go" in debug_lines[1]

    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.INFO and "quota_ordering" in record.getMessage()
    ], "ordering decisions must not be logged at INFO"


# ---------------------------------------------------------------------------
# Kill switch / dry-run — the predicate is the single review point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_kill_switch_keeps_config_order(caplog):
    """``quota_surge_enabled: false`` restores the exact pre-feature chain."""
    model_config = _two_account_chain()
    config = _config(quota_surge_enabled=False)
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    assert _quota_ordering_active(config) is False
    with caplog.at_level(logging.DEBUG, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
    assert not [r for r in caplog.records if "quota_ordering" in r.getMessage()]


@pytest.mark.asyncio
async def test_dry_run_logs_but_does_not_reorder(caplog):
    """Dry-run computes and logs the ranking but keeps the config order."""
    model_config = _two_account_chain()
    config = _config(quota_surge_dry_run=True)
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    assert _quota_ordering_active(config) is False
    with caplog.at_level(logging.DEBUG, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
    decisions = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and "quota_ordering" in r.getMessage()
    ]
    assert len(decisions) == 2
    assert all("applied=False" in line for line in decisions)


# ---------------------------------------------------------------------------
# Purity — the caller's provider list is never mutated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranked_providers_returns_copy_without_mutating_config():
    """Ordering works on a copy: the configured chain is left untouched."""
    model_config = _two_account_chain()
    config = _config()
    original = list(model_config["providers"])
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_percent=88),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=5),
        },
    )

    ranked = _quota_ranked_providers(model_config, config)
    assert ranked is not model_config["providers"]
    assert _names(ranked) == ["opencode-go-3", "opencode-go"]
    assert model_config["providers"] == original
    assert _names(model_config["providers"]) == ["opencode-go", "opencode-go-3"]
