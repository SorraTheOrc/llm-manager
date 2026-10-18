"""Tests for the OpenCode Go account-tier attribution in X-Resolved-Model
(LP-0MVDUW7OB006UT62).

The three OpenCode Go accounts (``opencode-go-3``, ``opencode-go-2``,
``opencode-go``) share ``provider: opencode-go`` and historically carried no
``display_name``, so the X-Resolved-Model header collapsed all of them to the
bare brand ``opencode-go/<model>``. Operators could not tell which account
tier served a request.

Each account now carries a distinct ``display_name`` (``OpenCode Go 3`` /
``OpenCode Go 2`` / ``OpenCode Go``), and the compact summarizer entry carries
``OpenCode Go Compact``, matching the established ``MunitoAI Free`` convention
(LP-0MVCQ36QV009QX8K). The change is display-only: ``provider`` (which feeds
failure-domain, cooldown and usage-limit keying) is untouched.

These tests assert the *resolved* runtime config through the production merge
path (base ``config.yaml`` and the merged fast/cheap profiles), so a tier
cannot silently drop out of a profile or drift in the chain order, and that
the resolved-model header values are pairwise distinct.
"""

import pytest
import yaml
from proxy.mode import proxy_dir
from proxy.provider import (
    _build_resolved_model_value,
    _entry_cooldown_key,
    _failure_domain_key,
    _resolve_display_provider_name,
    _usage_limit_account_key,
)
from tests.config_test_utils import get_merged_config, load_profile

#: Expected display strings for the three OpenCode Go account tiers.
OPENCODE_GO_TIERS = {
    "opencode-go-3": "OpenCode Go 3",
    "opencode-go-2": "OpenCode Go 2",
    "opencode-go": "OpenCode Go",
}
OPENCODE_GO_COMPACT_DISPLAY = "OpenCode Go Compact"

#: The chains that carry the three opencode-go accounts.
CHAINS = ("plan", "author", "code")


def _providers(cfg: dict, model: str) -> list[dict]:
    return cfg["models"][model]["providers"]


def _opencode_entry(providers: list[dict], name: str) -> dict:
    entry = next((p for p in providers if p["name"] == name), None)
    assert entry is not None, f"chain must retain provider entry {name!r}"
    return entry


@pytest.mark.parametrize("model", CHAINS)
def test_base_chain_opencode_entries_carry_distinct_display_names(model):
    """config.yaml carries tier-distinct display_name on each account.

    AC1: each account reports a distinct label; the provider brand that feeds
    failure-domain/cooldown/usage-limit keying is unchanged.
    """
    providers = _providers(load_profile("config.yaml"), model)
    for name, expected in OPENCODE_GO_TIERS.items():
        entry = _opencode_entry(providers, name)
        assert entry["display_name"] == expected, (
            f"{model}/{name} must carry display_name {expected!r}"
        )
        assert entry["provider"] == "opencode-go", (
            f"{model}/{name} must keep the shared opencode-go brand"
        )


@pytest.mark.parametrize("mode", ["fast", "cheap"])
@pytest.mark.parametrize("model", CHAINS)
def test_merged_chain_opencode_entries_carry_distinct_display_names(mode, model):
    """The tier labels survive the mode merge (fast overlay + base inheritance).

    AC5: the display_name fields are present in the merged runtime config used
    by both profiles.
    """
    providers = _providers(get_merged_config(mode), model)
    for name, expected in OPENCODE_GO_TIERS.items():
        entry = _opencode_entry(providers, name)
        assert entry["display_name"] == expected
        assert entry["provider"] == "opencode-go"


def test_fast_overlay_declares_display_names_directly():
    """config-fast.yaml repeats the entries in full, so it must carry the labels.

    AC5: the overlay file itself (not just the base) declares display_name.
    Lists replace the base wholesale in deep_merge, so a label missing here
    would be missing from the merged fast config too.
    """
    with open(proxy_dir() / "config-fast.yaml") as fh:
        overlay = yaml.safe_load(fh)
    for model in CHAINS:
        providers = overlay["models"][model]["providers"]
        entries = {
            p["name"]: p for p in providers if str(p["name"]).startswith("opencode")
        }
        for name, display in OPENCODE_GO_TIERS.items():
            assert entries[name]["display_name"] == display, (
                f"config-fast.yaml {model}/{name} must carry display_name {display!r}"
            )
    compact_entries = {
        p["name"]: p for p in overlay["models"]["compact"]["providers"]
    }
    compact = compact_entries["opencode-go-compact"]
    assert compact["display_name"] == OPENCODE_GO_COMPACT_DISPLAY


@pytest.mark.parametrize("mode", ["base", "fast", "cheap"])
def test_compact_entry_has_distinct_display_name(mode):
    """The compact summarizer entry is not reported as the bare opencode-go brand.

    AC2: ``opencode-go-compact`` resolves on its own display label in every
    profile.
    """
    cfg = load_profile("config.yaml") if mode == "base" else get_merged_config(mode)
    entry = _opencode_entry(cfg["models"]["compact"]["providers"], "opencode-go-compact")
    assert entry["display_name"] == OPENCODE_GO_COMPACT_DISPLAY
    assert entry["provider"] == "opencode-go"


def test_opencode_resolved_model_values_are_pairwise_distinct():
    """AC1: X-Resolved-Model resolves to distinct per-account display strings.

    The three plan-chain accounts (and the compact entry) must never resolve
    to the same header value.
    """
    fast = get_merged_config("fast")
    plan = _providers(fast, "plan")
    values = [
        _build_resolved_model_value(_opencode_entry(plan, name))
        for name in ("opencode-go-3", "opencode-go-2", "opencode-go")
    ]
    compact = _opencode_entry(fast["models"]["compact"]["providers"], "opencode-go-compact")
    values.append(_build_resolved_model_value(compact))

    assert values[0] == "OpenCode Go 3/deepseek-v4.1-flash"
    assert values[1] == "OpenCode Go 2/deepseek-v4.1-flash"
    assert values[2] == "OpenCode Go/deepseek-v4.1-flash"
    assert values[3] == "OpenCode Go Compact/muse-spark-1.3-contributor"
    assert len(set(values)) == len(values), (
        "All four OpenCode Go entries must resolve to pairwise-distinct values"
    )


def test_resolve_display_provider_name_priority_with_opencode():
    """AC3: display_name -> provider -> name precedence is unchanged."""
    assert _resolve_display_provider_name(
        {"display_name": "OpenCode Go 3", "provider": "opencode-go", "name": "opencode-go-3"}
    ) == "OpenCode Go 3"
    # Without display_name the shared brand resolves (pre-fix behaviour).
    assert _resolve_display_provider_name(
        {"provider": "opencode-go", "name": "opencode-go-3"}
    ) == "opencode-go"


def test_display_name_does_not_affect_opencode_keying():
    """AC4: adding display_name is display-only for the OpenCode Go entries.

    Failure-domain keys, usage-limit account keys and cooldown keys must be
    byte-identical with and without the field, because ``provider`` (the
    brand) is what feeds them.
    """
    cfg = {
        "name": "opencode-go-3",
        "type": "remote",
        "provider": "opencode-go",
        "display_name": "OpenCode Go 3",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_3_API_KEY",
        "forward_session_headers": False,
        "model": "deepseek-v4.1-flash",
    }
    cfg_without_display = {k: v for k, v in cfg.items() if k != "display_name"}
    assert _failure_domain_key(cfg) == _failure_domain_key(cfg_without_display)
    assert _usage_limit_account_key(cfg) == _usage_limit_account_key(cfg_without_display)
    assert _entry_cooldown_key(cfg) == _entry_cooldown_key(cfg_without_display)


def test_munitoai_tiers_remain_distinct():
    """AC3: no regression — MunitoAI free/paid stay distinct (LP-0MVCQ36QV009QX8K)."""
    providers = _providers(get_merged_config("fast"), "plan")
    free = next(p for p in providers if p["name"] == "munitoai-free")
    paid = next(p for p in providers if p["name"] == "munitoai")
    free_value = _build_resolved_model_value(free)
    paid_value = _build_resolved_model_value(paid)
    assert free_value == "MunitoAI Free/deepseek-ai/deepseek-v4-flash"
    assert paid_value == "MunitoAI/deepseek-ai/deepseek-v4-flash"
    assert free_value != paid_value
