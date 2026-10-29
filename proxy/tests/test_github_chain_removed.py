"""The unused ``github`` model chain is gone (LP-0MVMAX4KA003JPE6).

The shipped config used to declare a ``github`` model chain with a single
remote entry, ``github-primary`` (``https://models.inference.ai.azure.com``),
aliased ``github-*``. It carried zero traffic in every available proxy log and
declared no ``model`` (the client chose the upstream model), so nothing in this
deployment could exercise it. Removing it keeps the routing surface honest.

These tests exercise the removal through the public config/alias API — they
load the shipped ``config.yaml`` and resolve aliases the way the request path
does — so the dead entry cannot be reintroduced unnoticed, and so a removal
that leaves a dangling alias or provider reference behind fails loudly.

Covers:

- AC1/AC4 — the ``github`` chain, its ``github-primary`` entry and its
  ``github-*`` alias are absent from the shipped config.
- AC2 — the shipped config still loads (validators included) and no alias or
  provider reference dangles: every model keeps a non-empty chain for every
  priority, aliases resolve unambiguously, and provider entry names are unique
  within their model.
"""

import pytest
from config_test_utils import load_profile
from proxy.lifecycle import lookup_model_config
from proxy.priority import provider_chain_for_priority
from proxy.utils import load_config

#: The removed model chain and its single provider entry / brand.
GITHUB_MODEL = "github"
GITHUB_ENTRY = "github-primary"
GITHUB_BRAND = "github"
GITHUB_ENDPOINT = "models.inference.ai.azure.com"

#: Requests that the removed ``github-*`` alias used to resolve to the chain.
GITHUB_ALIAS_REQUESTS = (
    "github",
    "github-4o",
    "github-gpt-4o",
    "github-o1-preview",
)

#: Chains that must survive the removal untouched.
SURVIVING_MODELS = ("embed", "private", "plan", "author", "code", "compact")

PRIORITIES = ("low", "medium", "high")


@pytest.fixture(scope="module")
def models() -> dict:
    """The shipped ``models`` mapping, exactly as the request path sees it."""
    return load_profile("config.yaml")["models"]


class TestGithubChainAbsent:
    """AC1/AC4 — the dead chain cannot be reached or silently reintroduced."""

    def test_no_github_model_is_declared(self, models):
        assert GITHUB_MODEL not in models

    def test_no_provider_entry_or_brand_remains(self, models):
        offenders = [
            f"{model}:{entry.get('name')}"
            for model, model_cfg in models.items()
            for entry in model_cfg.get("providers") or []
            if entry.get("name") == GITHUB_ENTRY
            or str(entry.get("provider") or "").lower() == GITHUB_BRAND
            or GITHUB_ENDPOINT in str(entry.get("endpoint") or "")
        ]
        assert offenders == []

    @pytest.mark.parametrize("requested", GITHUB_ALIAS_REQUESTS)
    def test_no_github_alias_resolves_to_a_chain(self, models, requested):
        assert lookup_model_config(models, requested) is None

    def test_surviving_chains_are_still_declared(self, models):
        missing = [name for name in SURVIVING_MODELS if name not in models]
        assert missing == []


class TestShippedConfigHasNoDanglingReferences:
    """AC2 — the config still loads and every reference resolves."""

    def test_shipped_config_loads_through_validators(self):
        config = load_config()
        assert GITHUB_MODEL not in config["models"]
        for name in SURVIVING_MODELS:
            assert name in config["models"]

    def test_every_model_has_a_chain_for_every_priority(self, models):
        for model_name, model_cfg in models.items():
            for priority in PRIORITIES:
                chain = provider_chain_for_priority(model_cfg, priority)
                assert chain, f"{model_name} has no provider for {priority}"
                for entry in chain:
                    assert entry.get("name"), f"{model_name}: unnamed provider entry"

    def test_every_exact_alias_resolves_to_its_own_model(self, models):
        dangling = []
        for model_name, model_cfg in models.items():
            for alias in model_cfg.get("aliases") or []:
                if any(char in alias for char in "*?["):
                    continue
                if lookup_model_config(models, alias) != model_cfg:
                    dangling.append(f"{model_name}:{alias}")
        assert dangling == []

    def test_no_alias_is_claimed_by_two_models(self, models):
        claimed: dict[str, str] = {}
        ambiguous = []
        for model_name, model_cfg in models.items():
            for alias in model_cfg.get("aliases") or []:
                if alias in claimed:
                    ambiguous.append(f"{alias}: {claimed[alias]} vs {model_name}")
                claimed[alias] = model_name
        assert ambiguous == []

    def test_provider_entry_names_are_unique_within_a_model(self, models):
        duplicated = []
        for model_name, model_cfg in models.items():
            names = [entry.get("name") for entry in model_cfg.get("providers") or []]
            if len(names) != len(set(names)):
                duplicated.append(model_name)
        assert duplicated == []
