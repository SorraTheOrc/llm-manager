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
from proxy.provider import _build_resolved_model_value

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
    # Free-tier request budget (LP-0MVDQ1GMO00801PL): a small per-60s window,
    # so the tier is only used while it has capacity. Deliberately conservative
    # at 2/min (LP-0MVF19DKR008VCHO) — bursts spill to the paid tiers.
    "rate_limit_rpm": 2,
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
