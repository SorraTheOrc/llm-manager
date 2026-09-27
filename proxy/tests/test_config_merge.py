"""Unit tests for the layered config deep-merge helper (F1).

``config.yaml`` is the authoritative base config and the mode profiles
(``config-fast.yaml`` / ``config-cheap.yaml``) are overlays merged on top of
it. These tests pin the helper's semantics *before* the loader is wired to it
(F2) and before the mode files are deduplicated (F4):

* a new dict is returned (inputs are never mutated),
* nested dicts merge recursively at arbitrary depth,
* scalars, lists and ``None`` in the overlay replace the base value wholesale,
* keys present only in the base are inherited,
* keys present only in the overlay are added.
"""

import copy
import json
import logging
from pathlib import Path

import pytest
import yaml
from proxy.config_merge import deep_merge


class TestReturnValue:
    def test_returns_new_dict_not_base(self):
        base = {"a": 1}
        overlay = {"b": 2}
        merged = deep_merge(base, overlay)

        assert merged == {"a": 1, "b": 2}
        assert merged is not base
        assert merged is not overlay

    def test_empty_overlay_inherits_base(self):
        base = {"a": 1, "server": {"x": 10}}
        assert deep_merge(base, {}) == base

    def test_empty_base_yields_overlay(self):
        overlay = {"a": 1, "server": {"x": 10}}
        assert deep_merge({}, overlay) == overlay


class TestNestedMerge:
    def test_overlay_scalar_overrides_base_scalar(self):
        base = {"server": {"session_slot_pool_size": 1}}
        overlay = {"server": {"session_slot_pool_size": 3}}
        merged = deep_merge(base, overlay)

        assert merged["server"]["session_slot_pool_size"] == 3

    def test_missing_overlay_key_inherits_from_base(self):
        base = {"server": {"session_slot_pool_size": 1, "upstream_idle_timeout_seconds": 30}}
        overlay = {"server": {"session_slot_pool_size": 3}}
        merged = deep_merge(base, overlay)

        assert merged["server"]["session_slot_pool_size"] == 3
        assert merged["server"]["upstream_idle_timeout_seconds"] == 30

    def test_recursive_merge_at_arbitrary_depth(self):
        base = {"a": {"b": {"c": {"d": 1, "keep": True}}}}
        overlay = {"a": {"b": {"c": {"d": 2}}}}
        merged = deep_merge(base, overlay)

        assert merged["a"]["b"]["c"] == {"d": 2, "keep": True}

    def test_overlay_only_branch_is_added(self):
        base = {"server": {"a": 1}}
        overlay = {"server": {"b": 2}, "logging": {"level": "DEBUG"}}
        merged = deep_merge(base, overlay)

        assert merged["server"] == {"a": 1, "b": 2}
        assert merged["logging"] == {"level": "DEBUG"}

    def test_dict_over_scalar_replaces_whole_subtree(self):
        base = {"server": "unset"}
        overlay = {"server": {"a": 1}}
        merged = deep_merge(base, overlay)

        assert merged["server"] == {"a": 1}

    def test_scalar_over_dict_replaces_whole_subtree(self):
        base = {"server": {"a": 1}}
        overlay = {"server": None}
        merged = deep_merge(base, overlay)

        assert merged["server"] is None


class TestReplacementSemantics:
    def test_overlay_list_replaces_base_list_wholesale(self):
        base = {"models": {"qwen": {"providers": [{"name": "a"}, {"name": "b"}]}}}
        overlay = {"models": {"qwen": {"providers": [{"name": "c"}]}}}
        merged = deep_merge(base, overlay)

        assert merged["models"]["qwen"]["providers"] == [{"name": "c"}]

    def test_overlay_list_is_copied_not_aliased(self):
        overlay_list = [1, 2, 3]
        merged = deep_merge({}, {"values": overlay_list})

        assert merged["values"] == overlay_list
        assert merged["values"] is not overlay_list

    def test_none_overlay_value_replaces(self):
        base = {"a": 1, "b": 2}
        overlay = {"a": None}
        merged = deep_merge(base, overlay)

        assert merged == {"a": None, "b": 2}
        assert "a" in merged


class TestImmutability:
    def test_inputs_are_never_mutated(self):
        base = {"server": {"a": 1}, "models": {"m": {"providers": [1, 2]}}}
        overlay = {"server": {"b": 2}, "models": {"m": {"providers": [3]}}}
        base_snapshot = copy.deepcopy(base)
        overlay_snapshot = copy.deepcopy(overlay)

        deep_merge(base, overlay)

        assert base == base_snapshot
        assert overlay == overlay_snapshot

    def test_mutating_result_does_not_affect_inputs(self):
        base = {"server": {"a": 1}}
        overlay = {"server": {"b": 2}}

        merged = deep_merge(base, overlay)
        merged["server"]["a"] = 999
        merged["server"]["c"] = 3

        assert base == {"server": {"a": 1}}
        assert overlay == {"server": {"b": 2}}


class TestInputValidation:
    @pytest.mark.parametrize("base", [None, [], "text", 3])
    def test_non_dict_base_raises_type_error(self, base):
        with pytest.raises(TypeError):
            deep_merge(base, {"a": 1})

    @pytest.mark.parametrize("overlay", [None, [], "text", 3])
    def test_non_dict_overlay_raises_type_error(self, overlay):
        with pytest.raises(TypeError):
            deep_merge({"a": 1}, overlay)


# ---------------------------------------------------------------------------
# F2: load_config() merges the base config with the mode overlay
# ---------------------------------------------------------------------------


@pytest.fixture
def mode_file(tmp_path):
    """Temp path used as the persisted mode state file."""
    return tmp_path / ".mode"


def _read_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _base_config():
    from proxy.mode import proxy_dir

    return _read_yaml(proxy_dir() / "config.yaml")


def _select_mode(monkeypatch, mode_file, mode):
    """Persist *mode* and clear the env override for the no-path case."""
    from proxy import mode as mode_module

    monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
    monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
    mode_file.write_text(mode + "\n")


class TestLoadConfigMerge:
    """``load_config()`` with no path merges ``config.yaml`` with the
    mode overlay; an explicit path keeps today's raw single-file load."""

    def test_no_path_merges_overlay_onto_base(self, mode_file, monkeypatch):
        from proxy.mode import proxy_dir
        from proxy.utils import load_config

        _select_mode(monkeypatch, mode_file, "fast")
        cfg = load_config()

        overlay = _read_yaml(proxy_dir() / "config-fast.yaml")
        base = _base_config()

        # Overlay value wins (fast raises the idle timeout 30 -> 240).
        assert overlay["server"]["upstream_idle_timeout_seconds"] == 240
        assert cfg["server"]["upstream_idle_timeout_seconds"] == 240
        # Base-only keys survive because ``server`` is merged recursively,
        # not replaced wholesale.
        assert "startup_ramp" not in overlay["server"]
        assert cfg["server"]["startup_ramp"] == base["server"]["startup_ramp"]
        assert cfg["server"]["timeout_keep_alive"] == base["server"]["timeout_keep_alive"]

    def test_no_path_merge_inherits_base_only_keys_in_cheap_mode(
        self, mode_file, monkeypatch
    ):
        from proxy.utils import load_config

        _select_mode(monkeypatch, mode_file, "cheap")
        cfg = load_config()

        base = _base_config()
        assert cfg["server"]["session_slot_pool_size"] == 3  # overlay wins
        assert cfg["server"]["summarizer_disable_thinking"] == base["server"]["summarizer_disable_thinking"]
        assert cfg["server"]["upstream_activity_timeout_seconds"] == base["server"]["upstream_activity_timeout_seconds"]

    def test_explicit_path_does_not_merge(self, monkeypatch):
        from proxy.mode import proxy_dir
        from proxy.utils import load_config

        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        raw = _read_yaml(proxy_dir() / "config-fast.yaml")
        cfg = load_config(config_path=str(proxy_dir() / "config-fast.yaml"))

        assert cfg == raw
        assert "startup_ramp" not in cfg["server"]

    def test_missing_overlay_falls_back_to_base_with_warning(
        self, monkeypatch, tmp_path, caplog
    ):
        from proxy.utils import load_config

        missing = tmp_path / "does-not-exist.yaml"
        monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(missing))

        with caplog.at_level(logging.WARNING, logger="llama-proxy"):
            cfg = load_config()

        assert cfg == _base_config()
        assert any("does-not-exist.yaml" in r.getMessage() for r in caplog.records)

    def test_env_overlay_merges_onto_base(self, monkeypatch, tmp_path):
        from proxy.utils import load_config

        overlay_path = tmp_path / "custom-overlay.yaml"
        overlay_path.write_text(
            "server:\n  timeout_keep_alive: 7\n", encoding="utf-8"
        )
        monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(overlay_path))

        cfg = load_config()

        assert cfg["server"]["timeout_keep_alive"] == 7
        assert cfg["server"]["startup_ramp"] == _base_config()["server"]["startup_ramp"]

    def test_env_pointing_at_base_loads_the_file_once(self, monkeypatch):
        import proxy.utils as utils
        from proxy.mode import proxy_dir

        monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(proxy_dir() / "config.yaml"))

        # Read the expected base before patching the loader so the count only
        # observes load_config()'s own YAML reads.
        expected = _base_config()
        calls = {"n": 0}
        real_safe_load = yaml.safe_load

        def counting_safe_load(stream):
            calls["n"] += 1
            return real_safe_load(stream)

        monkeypatch.setattr(utils.yaml, "safe_load", counting_safe_load)

        cfg = utils.load_config()

        assert cfg == expected
        assert calls["n"] == 1  # base only: no second load of the same file

    def test_validation_runs_once_on_the_merged_config(
        self, mode_file, monkeypatch
    ):
        import proxy.utils as utils

        _select_mode(monkeypatch, mode_file, "fast")

        calls = []
        monkeypatch.setattr(utils, "_validate_prompt_configs", lambda cfg: calls.append("prompt"))
        monkeypatch.setattr(utils, "_validate_chain_hold_config", lambda cfg: calls.append("chain"))
        monkeypatch.setattr(utils, "_validate_compaction_config", lambda cfg: calls.append("compaction"))

        utils.load_config()

        assert calls == ["prompt", "chain", "compaction"]

    def test_returns_a_dict(self, mode_file, monkeypatch):
        from proxy.utils import load_config

        _select_mode(monkeypatch, mode_file, "cheap")
        assert isinstance(load_config(), dict)


# ---------------------------------------------------------------------------
# F3: golden equivalence regression (merge + the later F4 dedup)
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"

#: Server keys present in ``config.yaml`` but absent from both pre-dedup mode
#: files. Merging the base into fast/cheap intentionally inherits exactly
#: these (the "config.yaml authoritative" behavioural change).
BASE_ONLY_SERVER_KEYS = {
    "session_single_flight_duplicate_retry_after_seconds",
    "session_slot_availability_timeout_seconds",
    "sibling_fallback_cooldown_seconds",
    "sibling_fallback_threshold",
    "sibling_fallback_window_seconds",
    "startup_ramp",
    "summarizer_disable_thinking",
    "summarizer_reasoning_effort",
    "timeout_graceful_shutdown",
    "timeout_keep_alive",
    "upstream_activity_timeout_seconds",
    "upstream_empty_retry_timeout_seconds",
    "upstream_max_stream_duration_seconds",
}


def _fixture(name):
    with open(FIXTURES / name) as fh:
        return json.load(fh)


def _leaf_paths(obj, prefix=""):
    """Map every leaf of a nested config to its dotted path."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(_leaf_paths(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: obj}


def _get_path(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def _dedup_overlay(base, overlay):
    """Return *overlay* with every key-value pair equal to *base* removed.

    Mirrors the F4 dedup of the mode files: dicts are compared recursively
    (a partially-differing dict keeps only its differing keys), every other
    value (scalar, list, ``None``) is dropped when it equals the base value.
    """
    result = {}
    for key, value in overlay.items():
        if key in base:
            base_value = base[key]
            if isinstance(value, dict) and isinstance(base_value, dict):
                nested = _dedup_overlay(base_value, value)
                if nested:
                    result[key] = nested
                continue
            if value == base_value:
                continue
        result[key] = value
    return result


class TestGoldenMergedConfig:
    """Merge (and the later dedup) must be behaviour-preserving.

    ``merged_config_<mode>.json`` is a golden snapshot of the merged config
    taken *before* the mode files are deduplicated, and
    ``prededup_raw_<mode>.json`` is the corresponding pre-dedup overlay.
    Together they fix the expected merged surface so F4's dedup can be proven
    to change nothing.
    """

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_load_config_matches_golden(self, mode, mode_file, monkeypatch):
        from proxy.utils import load_config

        _select_mode(monkeypatch, mode_file, mode)
        assert load_config() == _fixture(f"merged_config_{mode}.json")

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_dedup_is_behaviour_preserving(self, mode):
        """Removing base-equal entries from the overlay cannot change the
        merge result (this is the gate F4 must not break)."""
        base = _base_config()
        raw = _fixture(f"prededup_raw_{mode}.json")
        golden = _fixture(f"merged_config_{mode}.json")

        assert deep_merge(base, _dedup_overlay(base, raw)) == golden

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_only_documented_base_keys_are_inherited(self, mode):
        raw = _fixture(f"prededup_raw_{mode}.json")
        golden = _fixture(f"merged_config_{mode}.json")

        inherited = set(golden["server"]) - set(raw["server"])
        assert inherited == BASE_ONLY_SERVER_KEYS
        # No top-level section is introduced by the merge.
        assert set(golden) - set(raw) == set()

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_inherited_values_match_base(self, mode):
        base = _base_config()
        golden = _fixture(f"merged_config_{mode}.json")

        for key in BASE_ONLY_SERVER_KEYS:
            assert golden["server"][key] == base["server"][key]

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_overlay_values_still_win(self, mode):
        raw = _fixture(f"prededup_raw_{mode}.json")
        golden = _fixture(f"merged_config_{mode}.json")

        for path, value in _leaf_paths(raw).items():
            assert _get_path(golden, path) == value


def _find_base_duplicates(base, overlay, prefix=""):
    """Dotted paths of overlay leaves that duplicate the base value.

    Recurses through mappings; a list is treated as a leaf (it replaces the
    base list wholesale, so its inner entries may legitimately repeat base).
    """
    duplicates = []
    for key, value in overlay.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in base:
            continue
        base_value = base[key]
        if isinstance(value, dict) and isinstance(base_value, dict):
            duplicates.extend(_find_base_duplicates(base_value, value, path))
        elif value == base_value:
            duplicates.append(path)
    return duplicates


class TestDeduplicatedModeFiles:
    """F4: the mode files retain only values that differ from ``config.yaml``."""

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_no_base_duplicates_remain(self, mode):
        from proxy.mode import proxy_dir

        base = _base_config()
        overlay = _read_yaml(proxy_dir() / f"config-{mode}.yaml")

        assert _find_base_duplicates(base, overlay) == []

    @pytest.mark.parametrize(
        "mode,expected_top_keys",
        [("fast", {"models", "server"}), ("cheap", {"server"})],
    )
    def test_residue_shape(self, mode, expected_top_keys):
        from proxy.mode import proxy_dir

        overlay = _read_yaml(proxy_dir() / f"config-{mode}.yaml")
        assert set(overlay) == expected_top_keys

    @pytest.mark.parametrize(
        "mode,expected_server_keys",
        [
            (
                "fast",
                {
                    "contention_queue_policy",
                    "contention_queue_max_wait_seconds",
                    "contention_queue_max_depth",
                    "mode_switch_drain",
                    "upstream_idle_timeout_seconds",
                },
            ),
            (
                "cheap",
                {
                    "local_large_context_cold_cache_threshold",
                    "local_large_context_economic_bypass_serves_local",
                    "session_slot_pool_size",
                    "contention_queue_policy",
                    "contention_queue_max_wait_seconds",
                    "contention_queue_max_depth",
                    "mode_switch_drain",
                    "upstream_idle_timeout_seconds",
                },
            ),
        ],
    )
    def test_residue_server_keys(self, mode, expected_server_keys):
        from proxy.mode import proxy_dir

        overlay = _read_yaml(proxy_dir() / f"config-{mode}.yaml")
        assert set(overlay["server"]) == expected_server_keys

    def test_cheap_models_section_is_gone(self):
        """cheap's models are identical to base and are inherited."""
        from proxy.mode import proxy_dir

        overlay = _read_yaml(proxy_dir() / "config-cheap.yaml")
        assert "models" not in overlay


class TestLoadMergedHelper:
    """``config_merge.load_merged`` is the dependency-light shell entry point."""

    def test_load_merged_merges_overlay_onto_base(self, tmp_path):
        from proxy.config_merge import load_merged

        base = tmp_path / "base.yaml"
        base.write_text("server:\n  a: 1\n  nested:\n    x: 1\n")
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text("server:\n  b: 2\n  nested:\n    y: 2\n")

        assert load_merged(base, overlay) == {
            "server": {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}}
        }

    def test_load_merged_empty_documents_are_dicts(self, tmp_path):
        from proxy.config_merge import load_merged

        base = tmp_path / "base.yaml"
        base.write_text("")
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text("server:\n  a: 1\n")

        assert load_merged(base, overlay) == {"server": {"a": 1}}

    def test_load_merged_missing_file_raises(self, tmp_path):
        from proxy.config_merge import load_merged

        with pytest.raises(FileNotFoundError):
            load_merged(tmp_path / "missing-base.yaml", tmp_path / "missing-overlay.yaml")


# ---------------------------------------------------------------------------
# F5: shared test helper for the merged config surface
# ---------------------------------------------------------------------------


class TestConfigTestUtilsHelper:
    """``proxy/tests/config_test_utils`` provides the merged runtime surface.

    Tests must assert against this helper rather than parsing the raw mode
    overlays, so a value deduplicated into the base can never silently drop
    out of a profile assertion.
    """

    @pytest.mark.parametrize("mode", ["fast", "cheap"])
    def test_get_merged_config_matches_golden(self, mode):
        from tests.config_test_utils import get_merged_config

        assert get_merged_config(mode) == _fixture(f"merged_config_{mode}.json")

    def test_load_profile_returns_raw_base(self):
        from tests.config_test_utils import load_profile

        assert load_profile("config.yaml") == _base_config()

    @pytest.mark.parametrize(
        "mode,name",
        [("fast", "config-fast.yaml"), ("cheap", "config-cheap.yaml")],
    )
    def test_load_profile_returns_merged_mode(self, mode, name):
        from tests.config_test_utils import get_merged_config, load_profile

        assert load_profile(name) == get_merged_config(mode)

    def test_overlay_specific_values_are_explicit(self):
        """Genuinely overlay-specific facts are still asserted exactly."""
        from tests.config_test_utils import get_merged_config

        assert get_merged_config("cheap")["server"]["session_slot_pool_size"] == 3
        assert get_merged_config("cheap")["server"][
            "local_large_context_cold_cache_threshold"
        ] == 42000
        assert get_merged_config("fast")["server"]["session_slot_pool_size"] == 1
        assert get_merged_config("fast")["server"]["contention_queue_max_depth"] == 3
