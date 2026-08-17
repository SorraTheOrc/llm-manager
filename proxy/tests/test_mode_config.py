"""Tests for mode-based config resolution and the mode config profiles.

Covers (LP-0MSLMYEEU002IBH6):
- load_config() precedence: LLAMA_PROXY_CONFIG env > mode-selected file >
  proxy/config.yaml default
- config-fast.yaml mirrors the current config.yaml day settings (3-slot,
  remote providers eligible)
- config-cheap.yaml is a 2-slot profile with the SAME models/provider
  chains as fast (remote providers enabled, LP-0MSMIPPJI007GU9N); it
  differs only in the local slot pool (2 vs 3) and the per-period
  ctx_size (262144 vs 131072, LP-0MSMZOAJW002UR2A)
- resolve_config_path() maps modes to the correct profile files
"""

import os

import pytest
import yaml
from proxy.utils import load_config

from proxy import mode as mode_module


@pytest.fixture
def mode_file(tmp_path):
    """Return a temp path to use as the mode state file."""
    return tmp_path / ".mode"


# ---------------------------------------------------------------------------
# load_config() / resolve_config_path() precedence
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_env_var_wins_over_mode(self, mode_file, monkeypatch):
        """LLAMA_PROXY_CONFIG explicitly overrides mode selection."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("cheap\n")
        monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(mode_module.proxy_dir() / "config.yaml"))
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 3

    def test_cheap_mode_selects_cheap_config(self, mode_file, monkeypatch):
        """Persisted cheap mode -> load_config reads config-cheap.yaml (2 slots)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("cheap\n")
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 2

    def test_fast_mode_selects_fast_config(self, mode_file, monkeypatch):
        """Persisted fast mode -> load_config reads config-fast.yaml (3 slots)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("fast\n")
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 3

    def test_no_mode_file_defaults_to_fast(self, mode_file, monkeypatch):
        """No persisted mode -> load_config reads the fast profile (default)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 3

    def test_resolve_config_path_precedence(self, mode_file, monkeypatch):
        """resolve_config_path follows env > mode > default."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)

        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        mode_file.write_text("cheap\n")
        assert resolve_config_path().name == "config-cheap.yaml"

        mode_file.write_text("fast\n")
        assert resolve_config_path().name == "config-fast.yaml"

        mode_file.unlink()
        assert resolve_config_path().name == "config-fast.yaml"  # absent -> fast

        monkeypatch.setenv("LLAMA_PROXY_CONFIG", str(mode_module.proxy_dir() / "config.yaml"))
        assert resolve_config_path().name == "config.yaml"


def resolve_config_path():
    """Local wrapper so the test can call mode.resolve_config_path() cleanly."""
    return mode_module.resolve_config_path()


# ---------------------------------------------------------------------------
# Home page mode display (ui._current_mode)
# ---------------------------------------------------------------------------


class TestHomePageModeDisplay:
    def test_current_mode_helper_returns_persisted_mode(self, mode_file, monkeypatch):
        """ui._current_mode() returns the persisted mode for the home page."""
        from proxy.ui import _current_mode

        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("cheap\n")
        assert _current_mode() == "cheap"

    def test_current_mode_helper_defaults_to_fast(self, mode_file, monkeypatch):
        """ui._current_mode() defaults to fast when nothing is persisted."""
        from proxy.ui import _current_mode

        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        assert _current_mode() == "fast"


# ---------------------------------------------------------------------------
# Config profile validation
# ---------------------------------------------------------------------------


def _load(name: str) -> dict:
    path = mode_module.proxy_dir() / name
    with open(path) as fh:
        return yaml.safe_load(fh)


class TestFastConfigProfile:
    def test_fast_config_is_3_slot(self):
        """config-fast.yaml keeps the 3-slot pool and 3/3 schedule."""
        cfg = _load("config-fast.yaml")
        srv = cfg["server"]
        assert srv["session_slot_pool_size"] == 3
        entries = srv["slot_schedule"]["entries"]
        assert [e["slots"] for e in entries] == [3, 3]

    def test_fast_config_has_remote_providers(self):
        """config-fast.yaml keeps the cloud provider cascade (day settings)."""
        cfg = _load("config-fast.yaml")
        remote = [
            (model, p["name"])
            for model, mc in cfg["models"].items()
            for p in mc.get("providers", [])
            if p.get("type") == "remote"
        ]
        assert remote, "fast config must retain remote (cloud) providers"
        # DeepSeek and OpenCode tiers are present.
        brands = {name.split("-")[0] for _, name in remote}
        assert "deepseek" in brands or any("deepseek" in n for _, n in remote)
        assert any("opencode" in n for _, n in remote)

    def test_fast_config_matches_current_default(self):
        """config-fast.yaml mirrors config.yaml (current day settings)."""
        base = _load("config.yaml")
        fast = _load("config-fast.yaml")
        assert fast["models"] == base["models"]
        assert fast["server"]["session_slot_pool_size"] == base["server"]["session_slot_pool_size"]
        assert fast["default_model"] == base["default_model"]


class TestCheapConfigProfile:
    def test_cheap_config_is_2_slot(self):
        """config-cheap.yaml uses a 2-slot pool, 2/2 schedule entries, each
        carrying the full-model ctx_size override (262144, LP-0MSMZOAJW002UR2A)."""
        cfg = _load("config-cheap.yaml")
        srv = cfg["server"]
        assert srv["session_slot_pool_size"] == 2
        entries = srv["slot_schedule"]["entries"]
        assert [e["slots"] for e in entries] == [2, 2]
        assert all(e.get("ctx_size") == 262144 for e in entries)

    def test_cheap_config_keeps_static_ctx_for_boot_catchup(self):
        """The cheap profile's static local_model_ctx_size stays at the
        models.ini base (131072) — deliberately NOT aligned with the schedule
        entries' 262144 — so the slot-scheduler boot catch-up fires and
        applies the per-period 256K override (LP-0MSMZOAJW002UR2A)."""
        cfg = _load("config-cheap.yaml")
        srv = cfg["server"]
        assert srv["local_model_ctx_size"] == 131072
        assert all(
            e.get("ctx_size", 0) != srv["local_model_ctx_size"]
            for e in srv["slot_schedule"]["entries"]
        )

    def test_cheap_config_has_remote_providers(self):
        """config-cheap.yaml keeps remote providers enabled (LP-0MSMIPPJI007GU9N)."""
        cfg = _load("config-cheap.yaml")
        remote = [
            (model, p["name"])
            for model, mc in cfg["models"].items()
            for p in mc.get("providers", [])
            if p.get("type") == "remote"
        ]
        assert remote, "cheap config must retain remote (cloud) providers"
        # DeepSeek and OpenCode tiers are present, as in fast mode.
        brands = {name.split("-")[0] for _, name in remote}
        assert "deepseek" in brands or any("deepseek" in n for _, n in remote)
        assert any("opencode" in n for _, n in remote)

    def test_cheap_config_keeps_local_models(self):
        """Local models (embed/plan/author/code) remain available and local-first."""
        cfg = _load("config-cheap.yaml")
        assert {"embed", "plan", "author", "code"} <= set(cfg["models"])
        assert "github" in cfg["models"]  # remote-only model restored (LP-0MSMIPPJI007GU9N)
        for model in ("plan", "author", "code"):
            providers = cfg["models"][model]["providers"]
            assert providers, f"{model} must have at least the local provider"
            assert providers[0]["type"] == "local", f"{model} must route local-first"

    def test_cheap_config_resolves_github_alias(self, monkeypatch):
        """A github-* request resolves in cheap mode via get_model_config (LP-0MSMIPPJI007GU9N)."""
        import proxy.server as server_module
        from proxy.lifecycle import get_model_config

        monkeypatch.setattr(server_module, "config", _load("config-cheap.yaml"))
        cfg = get_model_config("github-session")
        assert cfg is not None, "github-* must resolve in cheap mode"
        assert cfg["providers"][0]["type"] == "remote"

    def test_cheap_config_matches_fast_models(self):
        """cheap and fast expose identical models/provider chains (LP-0MSMIPPJI007GU9N)."""
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        assert cheap["models"] == fast["models"]

    def test_cheap_config_differs_from_fast_only_by_slot_pool_ctx_and_contention(self):
        """The only intended cheap-vs-fast server differences are the local
        slot pool (2 vs 3), the per-period ctx_size (262144 vs 131072), the
        per-mode contention-queue policy (cheap=queue vs fast=fallback,
        LP-0MSORQVK50012Q4D F2), and the cold-cache threshold (cheap 30000
        after revert vs fast 38000, LP-0MSOMVOPH004ATAK / LP-0MSRM54YO007YG0K
        AC7). Everything else (models, static ctx, warm threshold) is identical."""
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        cheap_srv = dict(cheap["server"])
        fast_srv = dict(fast["server"])
        cheap_srv["session_slot_pool_size"] = fast_srv["session_slot_pool_size"]
        cheap_srv["slot_schedule"] = fast_srv["slot_schedule"]
        # Cheap declares queue + caps; fast declares fallback (no caps).
        cheap_srv["contention_queue_policy"] = fast_srv["contention_queue_policy"]
        cheap_srv.pop("contention_queue_max_wait_seconds", None)
        cheap_srv.pop("contention_queue_max_depth", None)
        # Cold-cache threshold (LP-0MSOMVOPH004ATAK then reverted per
        # LP-0MSRM54YO007YG0K AC7): cheap 30000, fast 38000.
        cheap_srv.pop("local_large_context_cold_cache_threshold", None)
        fast_srv.pop("local_large_context_cold_cache_threshold", None)
        assert cheap_srv == fast_srv

        # The intended diffs, asserted explicitly:
        assert cheap["server"]["session_slot_pool_size"] == 2
        assert fast["server"]["session_slot_pool_size"] == 3
        cheap_entries = cheap["server"]["slot_schedule"]["entries"]
        fast_entries = fast["server"]["slot_schedule"]["entries"]
        assert [e["slots"] for e in cheap_entries] == [2, 2]
        assert [e["slots"] for e in fast_entries] == [3, 3]
        assert all(e.get("ctx_size") == 262144 for e in cheap_entries)
        assert all(e.get("ctx_size") == 131072 for e in fast_entries)
        assert cheap["server"]["contention_queue_policy"] == "queue"
        assert cheap["server"]["contention_queue_max_wait_seconds"] == 60
        assert cheap["server"]["contention_queue_max_depth"] == 4
        assert fast["server"]["contention_queue_policy"] == "fallback"
        assert cheap["server"]["local_large_context_cold_cache_threshold"] == 30000
        assert fast["server"]["local_large_context_cold_cache_threshold"] == 38000
        assert cheap["server"]["local_large_context_warm_cache_threshold"] == fast["server"]["local_large_context_warm_cache_threshold"]

    def test_cheap_config_matches_fast_on_local_ctx(self):
        """The local model context size is unchanged between profiles."""
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        assert cheap["server"]["local_model_ctx_size"] == fast["server"]["local_model_ctx_size"]
