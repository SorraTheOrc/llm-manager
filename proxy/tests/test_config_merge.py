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

import pytest
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
