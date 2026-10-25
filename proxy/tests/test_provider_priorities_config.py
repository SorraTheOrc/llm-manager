"""Per-provider ``priorities`` routing labels (LP-0MVGPZWY0001JD1D).

Priority-chain membership used to be declared twice: the model's ordered
``providers`` list *and* a per-model ``priority_routes`` map from priority to
an ordered list of provider **names**. This item deletes ``priority_routes``
and labels each provider entry with the priorities it may serve:

    chain(P) = [p for p in providers if P in p["priorities"]]

so membership is declared once, on the definition the operator is already
editing, and chain order is exactly ``providers`` order.

Covers:

- AC2 — every provider of ``plan``/``author``/``code``/``compact`` is labelled;
  ``munitoai`` / ``deepseek-flash`` declare exactly ``[medium, high]``.
- AC3 — the derived chains are byte-identical to the pre-change chains,
  frozen here as literals.
- AC5 — load-time validation rejects every malformed label shape and a stale
  ``priority_routes`` block (with a migration message).
- AC6 — order is ``providers`` order (no name lookup, no rank table).
- AC7 — per-model divergence: identical provider lists differing only in one
  entry's labels resolve to different chains.
- AC11 — a labelled model never silently widens an empty filtered chain to
  the full ``providers`` list (Q6 option (a): load-time rejection).
"""

import pytest
from config_test_utils import load_profile
from proxy.priority import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    effective_chain_for_request,
    provider_chain_for_priority,
)
from proxy.provider import validate_priority_config
from proxy.utils import _validate_config, load_config

# ---------------------------------------------------------------------------
# Frozen pre-change chains (AC3). These are the chains the shipped config
# resolved to *before* this refactor; the labels must reproduce them exactly.
# ---------------------------------------------------------------------------

CHAIN_MODELS = ("plan", "author", "code")

PLAN_PROVIDER_ORDER = [
    "local-qwen3",
    "munitoai-free",
    "opencode-go-3",
    "opencode-go-2",
    "opencode-go",
    "munitoai",
    "deepseek-flash",
]
PLAN_LOW_CHAIN = PLAN_PROVIDER_ORDER[:5]

COMPACT_PROVIDER_ORDER = ["opencode-go-compact", "deepseek-flash-compact"]

#: Models that declare no ``priorities`` at all: every provider stays eligible
#: for every priority (the documented legacy default, Q3).
UNLABELLED_MODELS = {
    "embed": ["local-embed"],
    "github": ["github-primary"],
    "private": ["local-qwen3"],
}

ALL_LABELLED_MODELS = CHAIN_MODELS + ("compact",)

ALL_PRIORITIES = (PRIORITY_LOW, PRIORITY_MEDIUM, PRIORITY_HIGH)


def _expected_chain(model_name: str, priority: str) -> list[str]:
    """Frozen expectation: the chain the pre-change config resolved to."""
    if model_name in CHAIN_MODELS:
        providers = PLAN_PROVIDER_ORDER
        return PLAN_LOW_CHAIN if priority == PRIORITY_LOW else providers
    if model_name == "compact":
        return COMPACT_PROVIDER_ORDER
    return UNLABELLED_MODELS[model_name]


def _model(providers) -> dict:
    return {"providers": providers}


def _prov(name, priorities=None, ptype="remote"):
    entry = {"name": name, "type": ptype}
    if priorities is not None:
        entry["priorities"] = priorities
    return entry


class _State:
    priority = None


class _Request:
    """Minimal request stub for :func:`effective_chain_for_request`."""

    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = _State()


# ---------------------------------------------------------------------------
# AC3 — behaviour-equivalent chains derived from the labels
# ---------------------------------------------------------------------------


class TestFrozenChainEquivalence:
    @pytest.mark.parametrize("model_name", ALL_LABELLED_MODELS + tuple(UNLABELLED_MODELS))
    @pytest.mark.parametrize("priority", ALL_PRIORITIES)
    def test_derived_chain_matches_pre_change_chain(self, model_name, priority):
        model = load_profile("config.yaml")["models"][model_name]
        chain = provider_chain_for_priority(model, priority)
        assert [e["name"] for e in chain] == _expected_chain(model_name, priority)

    @pytest.mark.parametrize("model_name", ALL_LABELLED_MODELS + tuple(UNLABELLED_MODELS))
    @pytest.mark.parametrize("priority", ALL_PRIORITIES)
    def test_chain_entries_are_the_live_provider_entries(self, model_name, priority):
        # The filter returns the model's own entries (same objects, same
        # keys), so no entry detail — budget, windows, context cap — can be
        # lost or rewritten by the routing change.
        model = load_profile("config.yaml")["models"][model_name]
        chain = provider_chain_for_priority(model, priority)
        by_name = {p["name"]: p for p in model["providers"]}
        for entry in chain:
            assert entry is by_name[entry["name"]]


# ---------------------------------------------------------------------------
# AC2 — labels on the shipped config
# ---------------------------------------------------------------------------


class TestLiveConfigLabels:
    @pytest.mark.parametrize("model_name", ALL_LABELLED_MODELS)
    def test_every_provider_declares_priorities(self, model_name):
        model = load_profile("config.yaml")["models"][model_name]
        for entry in model["providers"]:
            assert "priorities" in entry, f"{model_name}/{entry.get('name')}"

    @pytest.mark.parametrize("model_name", CHAIN_MODELS)
    def test_paid_tiers_are_medium_high_only(self, model_name):
        model = load_profile("config.yaml")["models"][model_name]
        labels = {e["name"]: e["priorities"] for e in model["providers"]}
        assert labels["munitoai"] == ["medium", "high"]
        assert labels["deepseek-flash"] == ["medium", "high"]

    @pytest.mark.parametrize("model_name", ALL_LABELLED_MODELS)
    def test_remaining_entries_serve_all_priorities(self, model_name):
        model = load_profile("config.yaml")["models"][model_name]
        for entry in model["providers"]:
            if entry["name"] in ("munitoai", "deepseek-flash"):
                continue
            assert entry["priorities"] == ["low", "medium", "high"], entry["name"]

    def test_unlabelled_models_stay_unlabelled(self):
        cfg = load_profile("config.yaml")
        for model_name in UNLABELLED_MODELS:
            for entry in cfg["models"][model_name]["providers"]:
                assert "priorities" not in entry, f"{model_name}/{entry['name']}"

    def test_shipped_config_passes_priority_validation(self):
        assert validate_priority_config(load_config()) == []


# ---------------------------------------------------------------------------
# AC6 — chain order is ``providers`` order (no name lookup, no rank table)
# ---------------------------------------------------------------------------


class TestChainOrder:
    def test_order_follows_providers_not_alphabetical_names(self):
        # Deliberately out of alphabetical order: the chain must keep the
        # ``providers`` order instead of sorting or ranking by name.
        model = _model(
            [
                _prov("zeta", [PRIORITY_LOW, PRIORITY_MEDIUM]),
                _prov("alpha", [PRIORITY_LOW, PRIORITY_MEDIUM]),
                _prov("mike", [PRIORITY_MEDIUM]),
            ]
        )
        assert [p["name"] for p in provider_chain_for_priority(model, PRIORITY_LOW)] == [
            "zeta",
            "alpha",
        ]
        assert [
            p["name"] for p in provider_chain_for_priority(model, PRIORITY_MEDIUM)
        ] == ["zeta", "alpha", "mike"]

    def test_name_resolver_is_gone(self):
        # AC6: the name-resolving helper (and its unknown-name warning path)
        # no longer exists on the module.
        from proxy import priority

        assert not hasattr(priority, "_resolve_chain_entries")

    def test_a_name_that_matches_nothing_is_simply_not_a_provider(self):
        # Name strings are no longer an indirection, so an unknown name can
        # neither be resolved, warned about, nor shorten a chain.
        model = _model([_prov("a", [PRIORITY_LOW]), _prov("b", [PRIORITY_LOW])])
        assert [p["name"] for p in provider_chain_for_priority(model, PRIORITY_LOW)] == [
            "a",
            "b",
        ]


# ---------------------------------------------------------------------------
# AC7 — per-model divergence without cross-model coupling
# ---------------------------------------------------------------------------


class TestPerModelDivergence:
    def test_identical_provider_lists_differing_only_in_labels_diverge(self):
        expensive_only = _model(
            [
                _prov("local", [PRIORITY_LOW, PRIORITY_HIGH], "local"),
                _prov("paid", [PRIORITY_HIGH]),
            ]
        )
        also_low = _model(
            [
                _prov("local", [PRIORITY_LOW, PRIORITY_HIGH], "local"),
                _prov("paid", [PRIORITY_LOW, PRIORITY_HIGH]),
            ]
        )
        assert [p["name"] for p in provider_chain_for_priority(expensive_only, PRIORITY_LOW)] == [
            "local"
        ]
        assert [p["name"] for p in provider_chain_for_priority(also_low, PRIORITY_LOW)] == [
            "local",
            "paid",
        ]
        # ... and the high chains stay identical: no cross-model coupling.
        assert [
            p["name"] for p in provider_chain_for_priority(expensive_only, PRIORITY_HIGH)
        ] == [p["name"] for p in provider_chain_for_priority(also_low, PRIORITY_HIGH)]

    def test_labelling_one_model_does_not_relabel_another(self):
        labelled = _model([_prov("a", [PRIORITY_HIGH])])
        unlabelled = _model([_prov("a")])
        assert provider_chain_for_priority(labelled, PRIORITY_LOW) == []
        assert [p["name"] for p in provider_chain_for_priority(unlabelled, PRIORITY_LOW)] == [
            "a"
        ]


# ---------------------------------------------------------------------------
# AC3/Q3 — unlabelled models keep every provider for every priority
# ---------------------------------------------------------------------------


class TestUnlabelledDefault:
    def test_unlabelled_model_returns_full_provider_list(self):
        model = _model([_prov("a", ptype="local"), _prov("b")])
        for priority in ALL_PRIORITIES:
            assert provider_chain_for_priority(model, priority) == model["providers"]


# ---------------------------------------------------------------------------
# AC5 / AC11 — load-time validation
# ---------------------------------------------------------------------------


def _labelled_cfg(providers, model_name="m"):
    return {"models": {model_name: {"providers": providers}}}


class TestValidatePriorityConfig:
    def test_valid_labelled_model_passes(self):
        cfg = _labelled_cfg(
            [
                _prov("a", ["low", "medium", "high"]),
                _prov("b", ["medium", "high"]),
            ]
        )
        assert validate_priority_config(cfg) == []

    def test_unlabelled_model_passes(self):
        assert validate_priority_config(_labelled_cfg([_prov("a")])) == []

    def test_non_list_priorities_rejected(self):
        for bad in ("low", {"low": True}, 3):
            problems = validate_priority_config(_labelled_cfg([_prov("a", bad)]))
            assert problems, bad
            assert any("list" in p.lower() for p in problems), problems

    def test_empty_list_rejected(self):
        problems = validate_priority_config(_labelled_cfg([_prov("a", [])]))
        assert any("empty" in p.lower() for p in problems), problems

    def test_unknown_priority_name_rejected(self):
        problems = validate_priority_config(
            _labelled_cfg([_prov("a", ["low", "urgent"])])
        )
        assert any("urgent" in p for p in problems), problems

    def test_duplicate_entry_rejected(self):
        problems = validate_priority_config(
            _labelled_cfg([_prov("a", ["low", "low"])])
        )
        assert any("duplicate" in p.lower() for p in problems), problems

    def test_partially_labelled_model_rejected(self):
        cfg = _labelled_cfg(
            [_prov("a", ["low", "medium", "high"]), _prov("b")]
        )
        problems = validate_priority_config(cfg)
        assert any("'b'" in p for p in problems), problems
        assert any("priorities" in p for p in problems), problems

    def test_labelled_model_missing_a_priority_is_rejected(self):
        # AC11 (Q6 option (a)): fail closed at load rather than silently
        # widening to paid tiers at 02:00 inside the cheap window.
        cfg = _labelled_cfg([_prov("a", ["high"])])
        problems = validate_priority_config(cfg)
        assert any("low" in p for p in problems), problems
        assert any("medium" in p for p in problems), problems

    def test_stale_priority_routes_block_rejected_with_migration_message(self):
        cfg = _labelled_cfg([_prov("a", ["low", "medium", "high"])])
        cfg["models"]["m"]["priority_routes"] = {"low": ["a"]}
        problems = validate_priority_config(cfg)
        message = " ".join(problems)
        assert "priority_routes" in message
        assert "priorities" in message
        assert "m" in message

    def test_validate_config_raises_on_stale_priority_routes(self):
        cfg = load_config()
        cfg["models"]["plan"]["priority_routes"] = {"low": ["local-qwen3"]}
        with pytest.raises(ValueError, match="priorit"):
            _validate_config(cfg)

    def test_validate_config_raises_on_partial_labels(self):
        cfg = load_config()
        del cfg["models"]["code"]["providers"][0]["priorities"]
        with pytest.raises(ValueError, match="priorit"):
            _validate_config(cfg)


# ---------------------------------------------------------------------------
# AC11 — a labelled model never silently widens a filtered chain
# ---------------------------------------------------------------------------


class TestNoSilentWidening:
    def test_empty_filtered_chain_is_empty_not_the_full_list(self):
        model = _model(
            [
                _prov("a", [PRIORITY_LOW]),
                _prov("b", [PRIORITY_LOW]),
            ]
        )
        assert provider_chain_for_priority(model, PRIORITY_HIGH) == []

    def test_effective_chain_does_not_widen_for_a_labelled_model(self):
        model = _model([_prov("a", [PRIORITY_LOW]), _prov("b", [PRIORITY_LOW])])
        server_cfg = {"priority_window": {"enabled": False}}
        chain = effective_chain_for_request(
            _Request({"X-Priority": PRIORITY_HIGH}), model, server_cfg
        )
        assert chain == []

    def test_effective_chain_still_returns_full_list_for_unlabelled_model(self):
        model = _model([_prov("a"), _prov("b")])
        server_cfg = {"priority_window": {"enabled": False}}
        chain = effective_chain_for_request(
            _Request({"X-Priority": PRIORITY_HIGH}), model, server_cfg
        )
        assert [p["name"] for p in chain] == ["a", "b"]
