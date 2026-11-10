"""Tests for the per-provider ``use_balance`` opt-in (LP-0MW46PAU4006SK7Z).

An OpenCode Go account may be configured in the console to fall over onto
prepaid credit once its monthly plan window is spent. ``use_balance:
true|false`` on a remote provider entry tells the quota gate which accounts
may do that. Only a real YAML boolean is honoured (fail safe); an omitted key
or any malformed value resolves to ``false`` with one WARN per entry — never
a load failure.

These tests cover (AC1/AC2/AC3/AC4/AC5/AC6/AC7):

- accept+parse: ``get_use_balance`` reads the flag from the provider-cfg dict;
- default: an omitted (or ``null``) ``use_balance`` yields ``false`` and
  routing behaviour is unchanged (regression);
- malformed: ``"yes"``, ``2``, a mapping … resolve to ``false``, log one WARN
  per entry (deduplicated), and never raise at load;
- opt-in config: the shipped ``config.yaml`` opts ``opencode-go`` in
  (``true``) in every chain it appears in and explicitly declares ``false``
  on ``opencode-go-2`` / ``opencode-go-3``;
- validation wiring: load-time validation scans the field (WARN side effect)
  and the shipped config validates clean.
"""

import pytest
import yaml
from proxy.priority import proxy_dir
from proxy.provider import (
    _USE_BALANCE_MALFORMED_WARNED,
    _USE_BALANCE_NON_QUOTA_NOTED,
    get_use_balance,
    validate_use_balance_config,
)
from proxy.utils import _validate_config
from tests.config_test_utils import get_merged_config, load_profile

#: The chains that carry the three opencode-go accounts (AC4).
CHAINS = ("plan", "author", "code")

#: The three account entries per chain (AC7).
OPENCODE_GO_NAMES = ("opencode-go", "opencode-go-2", "opencode-go-3")

#: Malformed values that must coerce to ``false`` (AC3): only a real YAML
#: boolean is honoured, so quoted strings, numbers and containers all fail.
MALFORMED_VALUES = [
    "yes",
    "true",  # quoted string, not the YAML boolean
    "false",
    "on",
    0,
    2,
    1.5,
    {},
    [],
]


@pytest.fixture(autouse=True)
def _isolate_use_balance_dedup():
    """Reset the one-WARN/debug-note-per-entry dedupes for every test."""
    _USE_BALANCE_MALFORMED_WARNED.clear()
    _USE_BALANCE_NON_QUOTA_NOTED.clear()
    yield
    _USE_BALANCE_MALFORMED_WARNED.clear()
    _USE_BALANCE_NON_QUOTA_NOTED.clear()


def _providers(cfg: dict, model: str) -> list[dict]:
    return cfg["models"][model]["providers"]


def _opencode_entry(providers: list[dict], name: str) -> dict:
    entry = next((p for p in providers if p["name"] == name), None)
    assert entry is not None, f"chain must retain provider entry {name!r}"
    return entry


class TestGetUseBalance:
    """AC1/AC2: parse, expose via the provider-cfg dict, default false."""

    def test_true_is_parsed_and_returned(self):
        entry = {"name": "opencode-go", "use_balance": True}
        assert get_use_balance(entry) is True

    def test_false_is_parsed_and_returned(self):
        entry = {"name": "opencode-go-2", "use_balance": False}
        assert get_use_balance(entry) is False

    @pytest.mark.parametrize(
        "entry",
        [
            {"name": "munitoai-free"},
            {"name": "opencode-go"},
            {},
            {"use_balance": None},  # explicit null == omitted
        ],
    )
    def test_omitted_or_null_defaults_false(self, entry):
        """AC2: an omitted ``use_balance`` yields ``false`` (regression).

        Routing behaviour is unchanged: every existing entry without the key
        resolves to the fail-safe default.
        """
        assert get_use_balance(entry) is False

    def test_non_dict_input_returns_false(self):
        assert get_use_balance(None) is False
        assert get_use_balance("opencode-go") is False


class TestMalformedUseBalance:
    """AC3: fail-safe coercion, one WARN per entry, no raise at load."""

    @pytest.mark.parametrize("raw", MALFORMED_VALUES)
    def test_malformed_resolves_false_with_one_warn(self, raw, caplog):
        entry = {"name": "opencode-go", "use_balance": raw}
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            assert get_use_balance(entry) is False
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(warns) == 1
        assert "opencode-go" in warns[0].getMessage()

    def test_malformed_warns_exactly_once_per_entry(self, caplog):
        """The router may read the flag on every request; one WARN only."""
        entry = {"name": "opencode-go", "use_balance": "yes"}
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            for _ in range(5):
                assert get_use_balance(entry) is False
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(warns) == 1

    def test_malformed_does_not_raise_at_load(self, caplog):
        """AC3/AC5: load validation WARNs but never fails startup."""
        cfg = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "opencode-go", "use_balance": "yes"},
                        {"name": "opencode-go-2", "use_balance": 2},
                        {"name": "munitoai-free", "use_balance": {}},
                    ]
                }
            }
        }
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            _validate_config(cfg)  # must not raise
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        # one WARN per *named* entry (AC3)
        names = {w.getMessage().split("'")[1] for w in warns}
        assert names == {"opencode-go", "opencode-go-2", "munitoai-free"}

    def test_validator_returns_empty_problem_list_and_warns(self, caplog):
        """The validator's contract: never a load-blocking problem."""
        cfg = {
            "models": {
                "plan": {
                    "providers": [{"name": "opencode-go", "use_balance": "yes"}]
                }
            }
        }
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            assert validate_use_balance_config(cfg) == []
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(warns) == 1

    def test_validate_then_getter_share_one_warn(self, caplog):
        """Load-time scan and runtime reads share the per-entry dedupe."""
        cfg = {
            "models": {
                "plan": {
                    "providers": [{"name": "opencode-go", "use_balance": "yes"}]
                }
            }
        }
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            validate_use_balance_config(cfg)
            assert get_use_balance(cfg["models"]["plan"]["providers"][0]) is False
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(warns) == 1

    def test_malformed_on_non_quota_entry_still_false_and_warns(self, caplog):
        entry = {"name": "munitoai-free", "use_balance": "yes"}
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            assert get_use_balance(entry) is False
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(warns) == 1


class TestNonQuotaEntries:
    """Scope: the flag is meaningful only for quota-managed (``opencode-*``)
    entries; elsewhere it is ignored with a debug note."""

    def test_true_on_quota_entry_is_meaningful(self, caplog):
        entry = {"name": "opencode-go", "provider": "opencode-go", "use_balance": True}
        with caplog.at_level("DEBUG", logger="llama-proxy.provider"):
            assert get_use_balance(entry) is True
        notes = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert notes == []

    def test_true_on_non_quota_entry_ignored_with_debug_note(self, caplog):
        entry = {
            "name": "munitoai-free",
            "provider": "MunitoAI",
            "type": "remote",
            "use_balance": True,
        }
        with caplog.at_level("DEBUG", logger="llama-proxy.provider"):
            assert get_use_balance(entry) is True  # value still read faithfully
        notes = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(notes) == 1
        assert "ignored" in notes[0].getMessage().lower()

    def test_non_quota_debug_note_deduplicated(self, caplog):
        entry = {
            "name": "munitoai-free",
            "provider": "MunitoAI",
            "use_balance": True,
        }
        with caplog.at_level("DEBUG", logger="llama-proxy.provider"):
            for _ in range(4):
                get_use_balance(entry)
        notes = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert len(notes) == 1


class TestShippedConfigDeclaration:
    """AC4/AC7: ``opencode-go`` opts in everywhere; ``opencode-go-2`` /
    ``opencode-go-3`` explicitly stay out."""

    @pytest.mark.parametrize("model", CHAINS)
    @pytest.mark.parametrize("cfg", ["base", "merged"])
    def test_opencode_go_opts_in_every_chain(self, model, cfg):
        loaded = (
            load_profile("config.yaml") if cfg == "base" else get_merged_config("fast")
        )
        providers = _providers(loaded, model)
        entry = _opencode_entry(providers, "opencode-go")
        assert "use_balance" in entry, (
            f"config.yaml {model}/opencode-go must declare use_balance (AC7)"
        )
        assert entry["use_balance"] is True, (
            f"{model}/opencode-go must opt in with use_balance: true (AC4)"
        )

    @pytest.mark.parametrize("model", CHAINS)
    def test_go2_go3_declared_false(self, model):
        loaded = load_profile("config.yaml")
        providers = _providers(loaded, model)
        for name in ("opencode-go-2", "opencode-go-3"):
            entry = _opencode_entry(providers, name)
            assert "use_balance" in entry, (
                f"config.yaml {model}/{name} must declare use_balance (AC7)"
            )
            assert entry["use_balance"] is False, (
                f"{model}/{name} must NOT opt in — use_balance must be false"
            )

    def test_all_three_accounts_declared_in_every_chain(self):
        base = load_profile("config.yaml")
        for model in CHAINS:
            decls = {
                p["name"]: p.get("use_balance")
                for p in _providers(base, model)
                if p["name"] in OPENCODE_GO_NAMES
            }
            assert set(decls) == set(OPENCODE_GO_NAMES), (
                f"{model}: every account entry must carry the field"
            )
            assert decls == {
                "opencode-go": True,
                "opencode-go-2": False,
                "opencode-go-3": False,
            }, f"{model}: only opencode-go opts in"

    def test_compact_entry_does_not_opt_in(self):
        base = load_profile("config.yaml")
        compact = _opencode_entry(base["models"]["compact"]["providers"], "opencode-go-compact")
        assert "use_balance" not in compact
        assert get_use_balance(compact) is False


class TestLoadTimeValidation:
    """AC5: the validator is wired into config load and the shipped config is
    clean (no warnings, no load failure)."""

    def test_shipped_config_validates_clean(self, caplog):
        cfg = load_profile("config.yaml")
        with caplog.at_level("WARNING", logger="llama-proxy.provider"):
            _validate_config(cfg)  # must not raise
        warns = [r for r in caplog.records if "use_balance" in r.getMessage()]
        assert warns == [], "shipped config carries no malformed use_balance"

    def test_validate_use_balance_config_returns_empty_for_shipped(self):
        assert validate_use_balance_config(load_profile("config.yaml")) == []
        assert validate_use_balance_config(get_merged_config("fast")) == []

    def test_validate_tolerates_missing_models(self):
        assert validate_use_balance_config({}) == []
        assert validate_use_balance_config(None) == []
        assert validate_use_balance_config({"models": "nonsense"}) == []
