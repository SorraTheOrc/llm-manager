"""Config-merge tests.

The fast/cheap mode-overlay system (``config-fast.yaml`` / ``config-cheap.yaml``)
was removed with the mode axis (LP-0MVDT94F0003VHUA AC4): ``config.yaml`` is
the single authoritative config and ``load_config()`` no longer consults a
persisted mode or the ``LLAMA_PROXY_CONFIG`` overlay selection. This suite
pins the deep-merge primitive (still used for custom overlays) and the
simplified load path.
"""

import json
import logging

import pytest
import yaml
from proxy.config_merge import deep_merge

# ---------------------------------------------------------------------------
# F1: deep_merge primitive
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_returns_new_dict_not_base(self):
        result = deep_merge({}, {})
        assert result == {}
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}
        assert result is not base

    def test_empty_overlay_inherits_base(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_empty_base_yields_overlay(self):
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_overlay_scalar_overrides_base_scalar(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_missing_overlay_key_inherits_from_base(self):
        assert deep_merge({"a": 1, "b": 2}, {"a": 9}) == {"a": 9, "b": 2}

    def test_recursive_merge_at_arbitrary_depth(self):
        base = {"server": {"timeout": 30, "nested": {"x": 1}}}
        overlay = {"server": {"timeout": 45, "nested": {"y": 2}}}
        assert deep_merge(base, overlay) == {
            "server": {"timeout": 45, "nested": {"x": 1, "y": 2}},
        }

    def test_overlay_only_branch_is_added(self):
        assert deep_merge({}, {"server": {"new": 1}}) == {"server": {"new": 1}}

    def test_dict_over_scalar_replaces_whole_subtree(self):
        assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}

    def test_scalar_over_dict_replaces_whole_subtree(self):
        assert deep_merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}

    def test_overlay_list_replaces_base_list_wholesale(self):
        base = {"server": {"list": [1, 2, 3]}}
        overlay = {"server": {"list": [9]}}
        assert deep_merge(base, overlay) == {"server": {"list": [9]}}

    def test_overlay_list_is_copied_not_aliased(self):
        overlay = {"server": {"list": [9]}}
        result = deep_merge({}, overlay)
        result["server"]["list"].append(10)
        assert overlay["server"]["list"] == [9]

    def test_none_overlay_value_replaces(self):
        assert deep_merge({"a": 1}, {"a": None}) == {"a": None}

    def test_inputs_are_never_mutated(self):
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        deep_merge(base, overlay)
        assert base == {"a": {"b": 1}}
        assert overlay == {"a": {"c": 2}}

    def test_mutating_result_does_not_affect_inputs(self):
        base = {"a": {"b": 1}}
        result = deep_merge(base, {})
        result["a"]["b"] = 99
        assert base == {"a": {"b": 1}}

    @pytest.mark.parametrize("bad", [None, [], "x", 5])
    def test_non_dict_base_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            deep_merge(bad, {})

    @pytest.mark.parametrize("bad", [None, [], "x", 5])
    def test_non_dict_overlay_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            deep_merge({}, bad)


# ---------------------------------------------------------------------------
# F2: load_config() — single authoritative base config
# ---------------------------------------------------------------------------


def _read_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _base_config():
    from proxy.priority import proxy_dir

    return _read_yaml(proxy_dir() / "config.yaml")


class TestLoadConfig:
    """``load_config()`` resolves the base ``config.yaml`` directly; the
    mode overlays and ``LLAMA_PROXY_CONFIG`` selection are gone."""

    def test_no_path_loads_base_config(self):
        from proxy.utils import load_config

        cfg = load_config()
        assert cfg == _base_config()
        assert cfg["server"]["session_slot_pool_size"] == 2

    def test_explicit_path_loads_that_file_raw(self, monkeypatch, tmp_path):
        from proxy.utils import load_config

        custom = tmp_path / "other.yaml"
        custom.write_text("server:\n  timeout_keep_alive: 7\n", encoding="utf-8")
        cfg = load_config(config_path=str(custom))
        assert cfg == {"server": {"timeout_keep_alive": 7}}

    def test_validation_runs_once(self, monkeypatch):
        import proxy.utils as utils

        calls = []
        monkeypatch.setattr(
            utils, "_validate_prompt_configs", lambda cfg: calls.append("prompt")
        )
        monkeypatch.setattr(
            utils, "_validate_chain_hold_config", lambda cfg: calls.append("chain")
        )
        monkeypatch.setattr(
            utils, "_validate_compaction_config", lambda cfg: calls.append("compaction")
        )

        utils.load_config()

        assert calls == ["prompt", "chain", "compaction"]

    def test_returns_a_dict(self):
        from proxy.utils import load_config

        assert isinstance(load_config(), dict)


class TestLoadMergedConfig:
    """``load_merged_config`` still honours a custom overlay (backward
    compatibility for consumers that supply one)."""

    def test_overlay_merges_onto_base(self, tmp_path):
        from proxy.utils import load_merged_config

        overlay_path = tmp_path / "custom-overlay.yaml"
        overlay_path.write_text(
            "server:\n  timeout_keep_alive: 7\n", encoding="utf-8"
        )
        cfg = load_merged_config(str(overlay_path))
        assert cfg["server"]["timeout_keep_alive"] == 7
        assert cfg["server"]["startup_ramp"] == _base_config()["server"]["startup_ramp"]

    def test_base_itself_returns_base(self):
        from proxy.priority import proxy_dir
        from proxy.utils import load_merged_config

        cfg = load_merged_config(str(proxy_dir() / "config.yaml"))
        assert cfg == _base_config()

    def test_missing_overlay_falls_back_to_base(self, tmp_path, caplog):
        from proxy.utils import load_merged_config

        missing = tmp_path / "does-not-exist.yaml"
        with caplog.at_level(logging.WARNING, logger="llama-proxy"):
            cfg = load_merged_config(str(missing))
        assert cfg == _base_config()
        assert any("does-not-exist.yaml" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Restart signature helper
# ---------------------------------------------------------------------------


class TestMergedConfigSha256:
    def test_deterministic_and_formatting_agnostic(self, tmp_path):
        from proxy.config_merge import merged_config_sha256

        base = tmp_path / "a.yaml"
        base.write_text("server:\n  x: 1\n", encoding="utf-8")
        overlay = tmp_path / "o.yaml"
        overlay.write_text("server:\n  y: 2\n", encoding="utf-8")

        h1 = merged_config_sha256(str(base), str(overlay))
        h2 = merged_config_sha256(str(base), str(overlay))
        assert h1 == h2
        assert len(h1) == 64

    def test_base_alone_hashes_stable(self, tmp_path):
        from proxy.config_merge import merged_config_sha256
        from proxy.priority import proxy_dir

        base = str(proxy_dir() / "config.yaml")
        assert merged_config_sha256(base, base) == merged_config_sha256(base, base)
