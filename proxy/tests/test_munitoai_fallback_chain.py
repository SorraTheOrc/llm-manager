"""Tests for the MunitoAI (``deepseek-ai/deepseek-v4-flash``) fallback tier
(LP-0MV5UPXKL002MSKB).

The MunitoAI tier is appended as the final remote fallback of the
``plan`` / ``author`` / ``code`` chains in the authoritative base config
(``config.yaml``) and in the fast overlay (``config-fast.yaml``); cheap
inherits the base chains. These tests assert the *resolved* runtime config
through the production merge path, so the tier cannot silently drop out of a
profile.

They also pin the two runtime facts the tier depends on: the resolved-model
header value derived from the entry, and resolution of the ``MUNITOAI_API_KEY``
environment variable from the ``munitoai`` entry in pi's ``auth.json``.
"""

import json

import pytest
from proxy.provider import _build_resolved_model_value
from tests.config_test_utils import get_merged_config, load_profile

#: The exact provider entry the chains must carry.
MUNITOAI_ENTRY = {
    "name": "munitoai",
    "type": "remote",
    "provider": "MunitoAI",
    "endpoint": "https://api.munito.ai/inference",
    "api_key_env": "MUNITOAI_API_KEY",
    "forward_session_headers": False,
    "model": "deepseek-ai/deepseek-v4-flash",
}

CHAINS = ("plan", "author", "code")


def _providers(cfg: dict, model: str) -> list[dict]:
    return cfg["models"][model]["providers"]


@pytest.mark.parametrize("model", CHAINS)
def test_base_chain_ends_with_munitoai(model):
    """config.yaml appends the MunitoAI tier last in every agent chain."""
    providers = _providers(load_profile("config.yaml"), model)
    assert providers[-1] == MUNITOAI_ENTRY


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_is_last_remote_tier(model):
    """MunitoAI is the final remote fallback (locals excluded)."""
    providers = _providers(load_profile("config.yaml"), model)
    remote = [p for p in providers if p.get("type") == "remote"]
    assert remote, f"{model} must retain remote fallbacks"
    assert remote[-1]["name"] == "munitoai"


@pytest.mark.parametrize("mode", ["fast", "cheap"])
@pytest.mark.parametrize("model", CHAINS)
def test_merged_profile_chain_ends_with_munitoai(mode, model):
    """The tier survives the mode merge (fast overlay + base inheritance)."""
    providers = _providers(get_merged_config(mode), model)
    assert providers[-1] == MUNITOAI_ENTRY


def test_munitoai_is_always_available():
    """No available_times window means the tier is eligible around the clock."""
    entry = _providers(load_profile("config.yaml"), "plan")[-1]
    assert entry["name"] == "munitoai"
    assert "available_times" not in entry


@pytest.mark.parametrize("model", CHAINS)
def test_munitoai_resolved_model_header(model):
    """The tier reports ``MunitoAI/<upstream model id>`` when it serves."""
    entry = _providers(get_merged_config("fast"), model)[-1]
    assert _build_resolved_model_value(entry) == "MunitoAI/deepseek-ai/deepseek-v4-flash"


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
