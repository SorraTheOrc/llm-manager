"""Tests for mode-based config resolution and the mode config profiles.

Covers (LP-0MSLMYEEU002IBH6):
- load_config() precedence: LLAMA_PROXY_CONFIG env > mode-selected file >
  proxy/config.yaml default
- config-fast.yaml mirrors the current config.yaml day settings (1-slot,
  remote providers eligible)
- config-cheap.yaml is a 3-slot profile with the SAME models/provider
  chains as fast (remote providers enabled, LP-0MSMIPPJI007GU9N); it
  differs only in the local slot pool (1 vs 3, the profile's single
  slot-count definition) and the contention/cold-cache caps
- no profile carries a ``slot_schedule`` or ``mode_schedule``: slot count
  is ``session_slot_pool_size`` and the mode-switch schedule lives in the
  standalone ``proxy/mode_schedule.yaml`` (LP-0MTZRM5HV0007S0V)
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
        assert cfg["server"]["session_slot_pool_size"] == 1

    def test_cheap_mode_selects_cheap_config(self, mode_file, monkeypatch):
        """Persisted cheap mode -> load_config reads config-cheap.yaml (3 slots)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("cheap\n")
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 3

    def test_fast_mode_selects_fast_config(self, mode_file, monkeypatch):
        """Persisted fast mode -> load_config reads config-fast.yaml (1 slot)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        mode_file.write_text("fast\n")
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 1

    def test_no_mode_file_defaults_to_fast(self, mode_file, monkeypatch):
        """No persisted mode -> load_config reads the fast profile (default)."""
        monkeypatch.setattr(mode_module, "mode_state_file", lambda: mode_file)
        monkeypatch.delenv("LLAMA_PROXY_CONFIG", raising=False)
        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 1

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


def _opencode_deepseek_names(models: dict) -> set:
    """Names of opencode-hosted deepseek provider entries across all models."""
    return {
        p["name"]
        for mc in models.values()
        for p in mc.get("providers", [])
        if p.get("name", "").startswith("opencode-") and "deepseek" in p["name"]
    }


def _strip_available_times(models: dict) -> dict:
    """Deep-copy models with available_times removed from every provider
    (used to compare profile chains modulo the intentional fast-vs-default
    timing difference, LP-0MSXPHTD9004DZ4P)."""
    import copy

    out = copy.deepcopy(models)
    for mc in out.values():
        for p in mc.get("providers", []):
            p.pop("available_times", None)
    return out


class TestFastConfigProfile:
    def test_fast_config_is_1_slot(self):
        """config-fast.yaml defines its slot count once (1) and has no
        time-based slot schedule (LP-0MTZRM5HV0007S0V)."""
        cfg = _load("config-fast.yaml")
        srv = cfg["server"]
        assert srv["session_slot_pool_size"] == 1
        assert "slot_schedule" not in srv
        assert "mode_schedule" not in srv

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
        # Profiles share the same models/chains modulo the intentional timing
        # difference: config.yaml times opencode-hosted deepseek, fast does
        # NOT (LP-0MSXPHTD9004DZ4P).
        assert _strip_available_times(fast["models"]) == _strip_available_times(base["models"])
        assert fast["server"]["session_slot_pool_size"] == base["server"]["session_slot_pool_size"]
        # Default times all opencode-hosted deepseek entries; fast leaves them untimed.
        assert _opencode_deepseek_names(base["models"]) == _opencode_deepseek_names(fast["models"])
        assert all(
            p.get("available_times") is not None
            for mc in base["models"].values()
            for p in mc.get("providers", [])
            if p.get("name", "").startswith("opencode-") and "deepseek" in p["name"]
        )
        assert all(
            "available_times" not in p
            for mc in fast["models"].values()
            for p in mc.get("providers", [])
            if p.get("name", "").startswith("opencode-") and "deepseek" in p["name"]
        )
        assert fast["default_model"] == base["default_model"]


class TestCheapConfigProfile:
    def test_cheap_config_is_3_slot(self):
        """config-cheap.yaml defines its slot count once (3) and has no
        time-based slot schedule (LP-0MTZRM5HV0007S0V; count per
        LP-0MU03AL730000B5W)."""
        cfg = _load("config-cheap.yaml")
        srv = cfg["server"]
        assert srv["session_slot_pool_size"] == 3
        assert "slot_schedule" not in srv
        assert "mode_schedule" not in srv

    def test_cheap_config_static_ctx_262144(self):
        """The cheap profile's static local_model_ctx_size is 262144
        (LP-0MTO8SZ8K0080RHT), giving the 2x262144 per-slot clamp."""
        cfg = _load("config-cheap.yaml")
        assert cfg["server"]["local_model_ctx_size"] == 262144

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
        """cheap and fast expose identical models/provider chains modulo the
        intentional timing difference: cheap times opencode-hosted deepseek
        (LP-0MSXPHTD9004DZ4P), fast does not."""
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        assert _strip_available_times(cheap["models"]) == _strip_available_times(fast["models"])
        assert _opencode_deepseek_names(cheap["models"]) == _opencode_deepseek_names(fast["models"])
        assert all(
            p.get("available_times") is not None
            for mc in cheap["models"].values()
            for p in mc.get("providers", [])
            if p.get("name", "").startswith("opencode-") and "deepseek" in p["name"]
        )

    def test_cheap_config_differs_from_fast_only_by_slot_pool_ctx_and_contention(self):
        """The only intended cheap-vs-fast server differences are the local
        slot pool (2 vs 3, the profile's single slot-count definition),
        local_model_ctx_size (both 262144 since LP-0MTO8SZ8K0080RHT), the
        per-mode contention-queue policy (both queue; fast smaller,
        LP-0MSORQVK50012Q4D F2 / LP-0MTQYIK4Z008XF2V), the cold-cache
        threshold (cheap 42000 raised from 38000, LP-0MSOMVOPH004ATAK
        / LP-0MSRM54YO007YG0K AC7 / LP-0MSY0V4ZO002ANPL / LP-0MT50SMU1005ZAD6;
        fast/default stays 38000 — cheap-only change, LP-0MT50WCCP000DU00),
        the economic-bypass recovery flag (cheap true, fast absent,
        LP-0MU5A4QBR003YJM0), and the persistence cap (cheap 126976 vs fast
        83285, each pinned to its mode's routing clamp, LP-0MTBTCB8D000OQ0C).
        Everything else (models, warm threshold) is identical. There is no
        slot_schedule in either profile (LP-0MTZRM5HV0007S0V)."""
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        cheap_srv = dict(cheap["server"])
        fast_srv = dict(fast["server"])
        cheap_srv["session_slot_pool_size"] = fast_srv["session_slot_pool_size"]
        # local_model_ctx_size now identical (262144) per LP-0MTO8SZ8K0080RHT.
        cheap_srv["local_model_ctx_size"] = fast_srv["local_model_ctx_size"]
        # Cheap declares queue + larger caps; fast declares queue + smaller caps
        # (LP-0MTQYIK4Z008XF2V: both modes queue, fast < cheap).
        cheap_srv["contention_queue_policy"] = fast_srv["contention_queue_policy"]
        cheap_srv.pop("contention_queue_max_wait_seconds", None)
        cheap_srv.pop("contention_queue_max_depth", None)
        fast_srv.pop("contention_queue_max_wait_seconds", None)
        fast_srv.pop("contention_queue_max_depth", None)
        # Cold-cache threshold (LP-0MSOMVOPH004ATAK; reverted per
        # LP-0MSRM54YO007YG0K AC7 then re-raised to 38000 per
        # LP-0MSY0V4ZO002ANPL, raised to 42000 for cheap only per
        # LP-0MT50SMU1005ZAD6 / LP-0MT50WCCP000DU00):
        # cheap 42000, fast 38000 (asymmetric — cheap-only change).
        cheap_srv.pop("local_large_context_cold_cache_threshold", None)
        fast_srv.pop("local_large_context_cold_cache_threshold", None)
        # Economic-bypass recovery flag (LP-0MU5A4QBR003YJM0): cheap enables
        # local recovery of the economic cold-cache bypass; fast omits the
        # key (default false).  Intended cheap-only difference.
        cheap_srv.pop("local_large_context_economic_bypass_serves_local", None)
        fast_srv.pop("local_large_context_economic_bypass_serves_local", None)
        # Persistence cap derived per profile from each mode's hard-routing
        # cap (LP-0MTBTCB8D000OQ0C → LP-0MTBOX45O005LD1S AC4): static 0
        # (derive) in both.
        cheap_srv.pop("session_slot_max_prompt_tokens", None)
        fast_srv.pop("session_slot_max_prompt_tokens", None)
        # Hard-routing-cap ratios DISABLED at 0 (LP-0MTLB1LK80098R43
        # revert of LP-0MTBOX45O005LD1S — each mode declares its own ratio
        # key now set to 0, so they match after pop for equality check).
        cheap_srv.pop("local_hard_routing_cap_ratio_cheap", None)
        fast_srv.pop("local_hard_routing_cap_ratio_fast", None)
        assert cheap_srv == fast_srv

        # The intended diffs, asserted explicitly:
        assert cheap["server"]["session_slot_pool_size"] == 3
        assert fast["server"]["session_slot_pool_size"] == 1
        assert "slot_schedule" not in cheap["server"]
        assert "slot_schedule" not in fast["server"]
        assert cheap["server"]["contention_queue_policy"] == "queue"
        # Caps tuned per LP-0MTF6EVLW007PEHN (T4 recommendation,
        # LP-0MTED3OFP006I7NO): wait 60→120, depth 4→8 (projected +35
        # dispatches/window, T3 a26bc66).  Fast mode declares a smaller queue
        # (LP-0MTQYIK4Z008XF2V: 3/45 vs cheap 8/120) so bursts spill to
        # remotes sooner during peak hours.
        assert cheap["server"]["contention_queue_max_wait_seconds"] == 120
        assert cheap["server"]["contention_queue_max_depth"] == 8
        assert fast["server"]["contention_queue_policy"] == "queue"
        assert fast["server"]["contention_queue_max_wait_seconds"] == 45
        assert fast["server"]["contention_queue_max_depth"] == 3
        assert fast["server"]["contention_queue_max_depth"] < cheap["server"]["contention_queue_max_depth"]
        assert fast["server"]["contention_queue_max_wait_seconds"] < cheap["server"]["contention_queue_max_wait_seconds"]
        assert cheap["server"]["local_large_context_cold_cache_threshold"] == 42000
        # Economic-bypass recovery flag (LP-0MU5A4QBR003YJM0): cheap enables
        # local recovery; fast omits the key (defaults to false).
        assert (
            cheap["server"]["local_large_context_economic_bypass_serves_local"]
            is True
        )
        assert (
            "local_large_context_economic_bypass_serves_local" not in fast["server"]
        )
        # Per-mode persistence caps derive from each profile's hard-routing
        # cap when enabled (LP-0MTBTCB8D000OQ0C → LP-0MTBOX45O005LD1S AC4)
        # but the cap is DISABLED in the revert (LP-0MTLB1LK80098R43 per
        # LP-0MTBTCK2I005MOTE NOT EFFECTIVE verdict): static 0 and hard cap 0.
        assert cheap["server"]["session_slot_max_prompt_tokens"] == 0
        assert fast["server"]["session_slot_max_prompt_tokens"] == 0
        # Per-mode hard-routing-cap ratios RETIRED (LP-0MTVXP7DG00613ZB AC3):
        # ``compaction_trigger_ratio`` is the single detection knob; the
        # legacy per-mode cap keys are removed from the live config surface.
        assert "local_hard_routing_cap_ratio_cheap" not in cheap["server"]
        assert "local_hard_routing_cap_ratio_fast" not in fast["server"]
        assert fast["server"]["local_large_context_cold_cache_threshold"] == 38000
        assert cheap["server"]["local_large_context_warm_cache_threshold"] == fast["server"]["local_large_context_warm_cache_threshold"]

    def test_cheap_config_matches_fast_on_local_ctx(self):
        """The local model context size: both fast and cheap inline
        262144 (LP-0MSY0SDAS0031Y7F + LP-0MTO8SZ8K0080RHT).
        """
        cheap = _load("config-cheap.yaml")
        fast = _load("config-fast.yaml")
        assert fast["server"]["local_model_ctx_size"] == 262144
        assert cheap["server"]["local_model_ctx_size"] == 262144


class TestSlotCountSingleSourceOfTruth:
    """Every mode profile defines its slot count exactly once
    (LP-0MTZRM5HV0007S0V)."""

    @pytest.mark.parametrize(
        "profile,expected_slots",
        [
            ("config.yaml", 1),
            ("config-fast.yaml", 1),
            ("config-cheap.yaml", 3),
        ],
    )
    def test_profile_defines_one_slot_count(self, profile, expected_slots):
        cfg = _load(profile)
        srv = cfg["server"]
        assert srv["session_slot_pool_size"] == expected_slots
        assert "slot_schedule" not in srv

    @pytest.mark.parametrize(
        "profile", ["config.yaml", "config-fast.yaml", "config-cheap.yaml"]
    )
    def test_profile_has_no_mode_schedule(self, profile):
        """Mode-switching policy lives in mode_schedule.yaml, not profiles."""
        assert "mode_schedule" not in _load(profile)["server"]

    def test_standalone_mode_schedule_file_exists_and_parses(self):
        path = mode_module.mode_schedule_file()
        assert path.is_file(), "proxy/mode_schedule.yaml must be tracked"
        schedule = mode_module.ModeScheduleConfig.from_file()
        assert schedule.enabled is True
        assert [(e.time.strftime("%H:%M"), e.mode) for e in schedule.entries] == [
            ("01:00", "cheap"),
            ("10:00", "fast"),
        ]
