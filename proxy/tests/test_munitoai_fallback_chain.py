"""Tests for the MunitoAI (``deepseek-ai/deepseek-v4-flash``) fallback tiers
(LP-0MV5UPXKL002MSKB, LP-0MV6ENYYH006XX0Z).

The MunitoAI tiers are the remote fallbacks of the ``plan`` / ``author`` /
``code`` chains: ``munitoai-free`` sits directly after the local Qwen3
provider (first remote tier), and ``munitoai`` sits directly before the
``deepseek-flash`` final tier, in both the authoritative base config
(``config.yaml``, the single authoritative config after the mode axis was
removed, LP-0MVDT94F0003VHUA AC4). These tests assert the *resolved* runtime config through the
production merge path, so the tier cannot silently drop out of a profile or
drift in the chain order.

They also pin the two runtime facts the tier depends on: the resolved-model
header value derived from the entry, and resolution of the ``MUNITOAI_API_KEY``
/ ``MUNITOAI_FREE_API_KEY`` environment variables from the ``munitoai`` /
``munitoai-free`` entries in pi's ``auth.json``.
"""

import json

import pytest
from config_test_utils import get_merged_config, load_profile
from proxy.provider import _build_resolved_model_value, _provider_context_limit

#: The exact provider entry the chains must carry right after local.
#: ``display_name`` marks the free tier (``MunitoAI Free``) so the
#: X-Resolved-Model header distinguishes it from the paid sibling
#: (LP-0MVCQ36QV009QX8K).
MUNITOAI_FREE_ENTRY = {
    "name": "munitoai-free",
    "type": "remote",
    "provider": "MunitoAI",
    "display_name": "MunitoAI Free",
    "endpoint": "https://api.munito.ai/inference",
    "api_key_env": "MUNITOAI_FREE_API_KEY",
    "forward_session_headers": False,
    "model": "deepseek-ai/deepseek-v4-flash",
    # Per-entry context ceiling (LP-0MVF9N31A0006VH5): the free tier's
    # rolling-60s input-token allowance (x-ratelimit-limit-input-tokens),
    # probed from the live endpoint 2026-10-19. Sourced, not guessed.
    "max_context_tokens": 150000,
    # Free-tier request budget (LP-0MVDQ1GMO00801PL): a small per-60s window,
    # so the tier is only used while it has capacity. Deliberately conservative
    # at 3/min (LP-0MVGGI4Y0003K4AZ) — bursts spill to the paid tiers.
    "rate_limit_rpm": 3,
    # Honour the upstream's x-ratelimit-reset-* headers on 429
    # (LP-0MVBHDNLB007ALLJ): MunitoAI sends no Retry-After, so the reset header
    # supplies the cooldown instead of a blind backoff guess.
    "respect_ratelimit_headers": True,
}

#: The exact (paid) munitoai entry that must sit directly before deepseek-flash.
MUNITOAI_ENTRY = {
    "name": "munitoai",
    "type": "remote",
    "provider": "MunitoAI",
    "endpoint": "https://api.munito.ai/inference",
    "api_key_env": "MUNITOAI_API_KEY",
    "forward_session_headers": False,
    "model": "deepseek-ai/deepseek-v4-flash",
    # Per-entry context ceiling (LP-0MVF9N31A0006VH5): the paid tier's
    # rolling-60s input-token allowance (x-ratelimit-limit-input-tokens),
    # probed from the live endpoint 2026-10-19. Same upstream model as the
    # free tier but a larger allowance, hence the deliberate higher limit.
    "max_context_tokens": 400000,
    # Honour x-ratelimit-reset-* headers on 429 (LP-0MVBHDNLB007ALLJ).
    "respect_ratelimit_headers": True,
}

CHAINS = ("plan", "author", "code")


def _providers(cfg: dict, model: str) -> list[dict]:
    return cfg["models"][model]["providers"]


def _first_opencode_index(providers: list[dict]) -> int:
    for index, provider in enumerate(providers):
        if provider.get("name", "").startswith("opencode"):
            return index
    raise AssertionError("chain must retain an opencode-go tier")


@pytest.mark.parametrize("model", CHAINS)
def test_base_chain_places_munitoai_free_after_local_qwen3(model):
    """config.yaml lists munitoai-free directly after the local Qwen3 provider."""
    providers = _providers(load_profile("config.yaml"), model)
    assert providers[0]["name"] == "local-qwen3"
    assert providers[1] == MUNITOAI_FREE_ENTRY


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_free_precedes_the_first_opencode_tier(model):
    """munitoai-free is the first remote tier, ahead of every opencode-go tier."""
    providers = _providers(load_profile("config.yaml"), model)
    munitoai_index = next(
        index for index, provider in enumerate(providers) if provider["name"] == "munitoai-free"
    )
    assert munitoai_index == 1
    assert munitoai_index < _first_opencode_index(providers)


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_sits_directly_before_deepseek_flash(model):
    """The paid munitoai tier sits immediately before the deepseek-flash tier."""
    providers = _providers(load_profile("config.yaml"), model)
    paid_index = next(
        index for index, provider in enumerate(providers) if provider["name"] == "munitoai"
    )
    assert providers[paid_index] == MUNITOAI_ENTRY
    assert paid_index >= _first_opencode_index(providers)
    assert paid_index + 1 == len(providers) - 1
    assert providers[paid_index + 1]["name"] == "deepseek-flash"


@pytest.mark.parametrize("model", CHAINS)
def test_deepseek_flash_is_the_final_tier(model):
    """The chain still ends with the direct deepseek-flash fallback."""
    providers = _providers(load_profile("config.yaml"), model)
    assert providers[-1]["name"] == "deepseek-flash"


@pytest.mark.parametrize("mode", ["fast", "cheap"])
@pytest.mark.parametrize("model", CHAINS)
def test_merged_profile_places_munitoai_free_after_local_qwen3(mode, model):
    """The position survives the mode merge (fast overlay + base inheritance)."""
    providers = _providers(get_merged_config(mode), model)
    assert providers[0]["name"] == "local-qwen3"
    assert providers[1] == MUNITOAI_FREE_ENTRY
    paid_index = next(
        index for index, provider in enumerate(providers) if provider["name"] == "munitoai"
    )
    assert providers[paid_index] == MUNITOAI_ENTRY
    assert paid_index + 1 == len(providers) - 1
    assert providers[paid_index + 1]["name"] == "deepseek-flash"


def test_munitoai_free_is_always_available():
    """No available_times window means the tier is eligible around the clock."""
    entry = _providers(load_profile("config.yaml"), "plan")[1]
    assert entry["name"] == "munitoai-free"
    assert "available_times" not in entry


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_resolved_model_header(model):
    """The tiers report tier-distinct resolved-model values when they serve.

    Free resolves to ``MunitoAI Free/<upstream model id>`` (via the entry's
    ``display_name``), the paid tier keeps ``MunitoAI/<upstream model id>``
    (LP-0MVCQ36QV009QX8K). The two strings must never be identical.
    """
    providers = _providers(get_merged_config("fast"), model)
    free_entry = providers[1]
    paid_entry = next(p for p in providers if p["name"] == "munitoai")
    free_value = _build_resolved_model_value(free_entry)
    paid_value = _build_resolved_model_value(paid_entry)
    assert free_value == "MunitoAI Free/deepseek-ai/deepseek-v4-flash"
    assert paid_value == "MunitoAI/deepseek-ai/deepseek-v4-flash"
    assert free_value != paid_value


@pytest.mark.parametrize("model", CHAINS)
def test_shipped_context_limit_field_is_a_positive_int(model):
    """A shipped ``max_context_tokens`` must be a positive integer.

    The per-entry context ceiling (LP-0MVF9MY0W002J2N3, parent
    LP-0MVF2DNCX003Z5CO) is optional: an entry that is deliberately ungated
    omits the key. When present, it must be a real positive ``int`` — a bool,
    string, zero or negative value would either silently disable the gate or
    mis-parse it at dispatch time.
    """
    for entry in _providers(load_profile("config.yaml"), model):
        if "max_context_tokens" not in entry:
            continue
        value = entry["max_context_tokens"]
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{entry['name']}: max_context_tokens must be an int, got {value!r}"
        )
        assert value > 0, f"{entry['name']}: use an absent key rather than {value!r}"


#: Sourced per-entry context ceilings shipped in ``proxy/config.yaml``
#: (LP-0MVF9N31A0006VH5). Provenance, recorded next to each value in the
#: config:
#:
#: * ``munitoai-free`` (150,000) and ``munitoai`` (400,000) are the tiers'
#:   documented rolling-60s **input-token allowances**, read from the live
#:   ``x-ratelimit-limit-input-tokens`` response header (probed 2026-10-19).
#:   The upstream model window (1,048,576) is far larger, so the allowance is
#:   the binding ceiling for a single dispatch.
#: * ``opencode-go*`` (1,000,000) is the model-registry context window for
#:   ``opencode-go/deepseek-v4.1-flash``; OpenCode Go publishes plan quota as
#:   percentages, not a per-minute token allowance, so the window is the only
#:   sourceable ceiling.
#: * ``deepseek-flash`` (1,048,576) is ``api.deepseek.com``'s own
#:   ``context_window`` for the model — the largest window in the chain, so
#:   the final tier stays effectively ungated (AC3).
SHIPPED_CONTEXT_LIMITS = {
    "munitoai-free": 150_000,
    "opencode-go-3": 1_000_000,
    "opencode-go-2": 1_000_000,
    "opencode-go": 1_000_000,
    "munitoai": 400_000,
    "deepseek-flash": 1_048_576,
}

#: Upstream context window of ``deepseek-ai/deepseek-v4-flash`` (per
#: api.deepseek.com ``/models``) — the model BOTH MunitoAI tiers serve, so a
#: MunitoAI ceiling below this value is an allowance-driven cap, not a model
#: limit.
DEEPSEEK_V4_FLASH_WINDOW = 1_048_576

#: Upstream context window of ``opencode-go/deepseek-v4.1-flash`` (model
#: registry) — the model all three ``opencode-go*`` accounts serve.
OPENCODE_V41_FLASH_WINDOW = 1_000_000

#: Upstream context window of the model the direct ``deepseek-flash`` entry
#: serves (``DeepSeek-V4.1-Flash``, per api.deepseek.com ``/models``).
DEEPSEEK_FLASH_WINDOW = 1_048_576

#: Upstream context window of ``muse-spark-1.3-contributor`` (model registry),
#: used by the ``compact`` chain's opencode-go tier.
MUSE_SPARK_WINDOW = 1_048_576


def _entry(providers: list[dict], name: str) -> dict:
    return next(provider for provider in providers if provider["name"] == name)


@pytest.mark.parametrize("model", CHAINS)
def test_shipped_context_limits_are_pinned(model):
    """Every gated remote tier ships its sourced ceiling exactly.

    The gate reads the value through ``provider._provider_context_limit``, so
    asserting the parsed limit (not just the raw YAML key) pins what dispatch
    will actually enforce. Drift in either the config or the reader fails here
    (LP-0MVF9N31A0006VH5 AC4).
    """
    providers = _providers(load_profile("config.yaml"), model)
    shipped = {
        entry["name"]: _provider_context_limit(entry)
        for entry in providers
        if entry["name"] in SHIPPED_CONTEXT_LIMITS
    }
    assert shipped == SHIPPED_CONTEXT_LIMITS


@pytest.mark.parametrize("model", CHAINS)
def test_every_remote_tier_in_the_chain_is_gated(model):
    """AC1: each remote tier in the chains declares a ceiling; local does not.

    The local tier keeps the existing local routing thresholds, so an absent
    limit there is correct — but every remote tier ships a sourced value.
    """
    for entry in _providers(load_profile("config.yaml"), model):
        limit = _provider_context_limit(entry)
        if entry["type"] == "remote":
            assert limit > 0, f"{entry['name']}: remote tier must be gated"
        else:
            assert limit == 0, f"{entry['name']}: local tier keeps local thresholds"


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_free_is_capped_by_its_per_minute_input_allowance(model):
    """The free tier's ceiling is its allowance, not the model's window.

    A single dispatch must not be able to consume more than one full minute of
    the free tier's input budget, so the shipped ceiling is far below the
    upstream window (the deliberate difference from the paid tier).
    """
    free_limit = _provider_context_limit(
        _entry(_providers(load_profile("config.yaml"), model), "munitoai-free")
    )
    assert free_limit < DEEPSEEK_V4_FLASH_WINDOW


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_paid_tier_allows_more_context_than_the_free_tier(model):
    """Same upstream model, larger allowance → larger deliberate ceiling."""
    providers = _providers(load_profile("config.yaml"), model)
    free_limit = _provider_context_limit(_entry(providers, "munitoai-free"))
    paid_limit = _provider_context_limit(_entry(providers, "munitoai"))
    assert paid_limit > free_limit
    assert paid_limit < DEEPSEEK_V4_FLASH_WINDOW


@pytest.mark.parametrize("model", CHAINS)
def test_opencode_tiers_share_the_same_context_limit(model):
    """The three opencode-go tiers serve the same model → the same ceiling.

    All three are capped at that model's upstream window (OpenCode Go exposes
    no per-minute token allowance to cap lower against), so an inconsistency
    between sibling accounts is a config error.
    """
    providers = _providers(load_profile("config.yaml"), model)
    limits = {
        _provider_context_limit(_entry(providers, name))
        for name in ("opencode-go-3", "opencode-go-2", "opencode-go")
    }
    assert limits == {OPENCODE_V41_FLASH_WINDOW}


@pytest.mark.parametrize("model", CHAINS)
def test_deepseek_flash_is_the_largest_window_destination(model):
    """AC3: the final tier holds the largest window so the chain never starves.

    An oversized prompt must still have a destination: the last tier's ceiling
    equals its upstream context window and is the largest ceiling in the
    chain, so no request can be capped out of every tier. It must also exceed
    the local warm-context threshold, otherwise a context the local model
    refuses would be refused remotely too.
    """
    cfg = load_profile("config.yaml")
    providers = _providers(cfg, model)
    limits = {p["name"]: _provider_context_limit(p) for p in providers}
    final_limit = limits["deepseek-flash"]
    assert final_limit == DEEPSEEK_FLASH_WINDOW
    assert final_limit == max(limits.values())
    assert final_limit > cfg["server"]["local_large_context_warm_cache_threshold"]


def test_compact_chain_context_limits_are_pinned():
    """The remote-only compact chain carries the same sourced ceilings."""
    providers = _providers(load_profile("config.yaml"), "compact")
    limits = {p["name"]: _provider_context_limit(p) for p in providers}
    assert limits == {
        "opencode-go-compact": MUSE_SPARK_WINDOW,
        "deepseek-flash-compact": DEEPSEEK_FLASH_WINDOW,
    }


def test_munitoai_api_key_resolves_from_pi_auth_json(tmp_path):
    """``MUNITOAI_API_KEY`` resolves from the ``munitoai`` auth.json entry.

    This mirrors ``start-proxy.sh``'s ``_api_key``-suffix-stripping lookup, so
    the operator's key is picked up without exporting the env var.
    """
    from unittest.mock import patch as mock_patch

    from proxy.proxy_remote import _try_pi_auth_json

    auth_data = {"munitoai": {"type": "api_key", "key": "sk-munito-test"}}
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        assert _try_pi_auth_json("MUNITOAI_API_KEY") == "sk-munito-test"
