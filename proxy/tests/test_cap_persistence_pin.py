"""Tests for LP-0MTE9HAF8008909G F3: pin persistence cap to the per-slot routing clamp.

The persistence cap must equal the per-slot routing clamp
(ctx // slots - headroom) for the live 262K schedule:

- 3 slots (fast): 262144 // 3 - 4096 = 83285
- 2 slots (cheap): 262144 // 2 - 4096 = 126976

Previously the dynamic derivation tried the hard-routing cap first
(70000 / 61440), leaving a gap where the router accepted a request as local
but persistence gated it out (F2: 38/48 sessions). The pin closes that gap
so accepted oversized sessions actually produce slot_save/slot_restore.

The GPU-wedge plan must stay unchanged:
  base 3.0s + 0.0015s/token, max 60s; cooldown 300s after 3 failures;
  skip-when-busy stays on.
"""

import pytest
from proxy.provider import effective_per_slot_threshold
from proxy.session import _build_slot_context, _slot_owners


def _body_for_tokens(n: int) -> dict:
    return {"messages": [{"role": "user", "content": "x" * (n * 8)}]}


def _make_config(**overrides):
    # Hard-routing caps DISABLED (LP-0MTLB1LK80098R43 revert of
    # LP-0MTBOX45O005LD1S per LP-0MTBTCK2I005MOTE NOT EFFECTIVE): 0 = dynamic
    # per-slot clamp, so persistence pins to 83285 / 126976.
    cfg = {
        "session_slot_save_path": "/tmp/slot-cache",
        "session_slot_pool_size": 3,
        "local_model_ctx_size": 262144,
        "session_slot_max_prompt_tokens": 0,
        "local_hard_routing_cap_ratio_fast": 0,
        "local_hard_routing_cap_ratio_cheap": 0,
        "warm_cache_threshold": 100000,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------ #
# Clamp constants
# ------------------------------------------------------------------ #

FAST_CLAMP = 83285  # 262144//3 - 4096
CHEAP_CLAMP = 126976  # 262144//2 - 4096


class TestPerSlotClampConstants:
    def test_fast_clamp_matches_resolved_context(self):
        assert effective_per_slot_threshold(262144, 3) == FAST_CLAMP

    def test_cheap_clamp_matches_resolved_context(self):
        assert effective_per_slot_threshold(262144, 2) == CHEAP_CLAMP


# ------------------------------------------------------------------ #
# Fast mode (3 slots): persistence gate pins at 83285
# ------------------------------------------------------------------ #

class TestFastCapPin:
    def test_under_clamp_persists(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=3)
        slot, fname, _ = _build_slot_context(cfg, "fast-under", _body_for_tokens(80000))
        assert slot is not None  # 80000 <= 83285

    def test_gap_session_now_persists(self):
        """F2 had 75000-token sessions gated out by the hard cap 70000.
        They must persist now that the cap is pinned to the clamp."""
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=3)
        slot, fname, _ = _build_slot_context(cfg, "fast-gap", _body_for_tokens(75000))
        assert slot is not None  # 75000 <= 83285

    def test_at_clamp_persists(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=3)
        slot, fname, _ = _build_slot_context(cfg, "fast-at-clamp", _body_for_tokens(FAST_CLAMP))
        assert slot is not None  # 83285 <= 83285

    def test_above_clamp_gated(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=3)
        slot, fname, _ = _build_slot_context(cfg, "fast-over", _body_for_tokens(86000))
        assert slot is None  # 86000 > 83285

    def test_static_cap_respected_when_set(self):
        """When a static cap is configured, it wins; only the 0 (dynamic)
        path is pinned."""
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=3, session_slot_max_prompt_tokens=70000)
        slot, fname, _ = _build_slot_context(cfg, "fast-static", _body_for_tokens(75000))
        assert slot is None  # gated by explicit 70000


# ------------------------------------------------------------------ #
# Cheap mode (2 slots): persistence gate pins at 126976
# ------------------------------------------------------------------ #

class TestCheapCapPin:
    def test_under_clamp_persists(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=2)
        slot, fname, _ = _build_slot_context(cfg, "cheap-under", _body_for_tokens(120000))
        assert slot is not None  # 120000 <= 126976

    def test_gap_session_now_persists(self):
        """A 100000-token session gated out by the hard cap 61440 must
        persist now that the cap is pinned to cheap's clamp."""
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=2)
        slot, fname, _ = _build_slot_context(cfg, "cheap-gap", _body_for_tokens(100000))
        assert slot is not None  # 100000 <= 126976

    def test_at_clamp_persists(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=2)
        slot, fname, _ = _build_slot_context(cfg, "cheap-at-clamp", _body_for_tokens(CHEAP_CLAMP))
        assert slot is not None  # 126976

    def test_above_clamp_gated(self):
        _slot_owners.clear()
        cfg = _make_config(session_slot_pool_size=2)
        slot, fname, _ = _build_slot_context(cfg, "cheap-over", _body_for_tokens(130000))
        assert slot is None  # 130000 > 126976


# ------------------------------------------------------------------ #
# GPU-wedge plan carried unchanged (AC2 requirement)
# ------------------------------------------------------------------ #

class TestGpuWedgePlanUnchanged:
    def test_adaptive_timeout_scaling(self):
        """Base 3.0s + 0.0015s/token, capped at 60s."""
        _slot_owners.clear()
        cfg = _make_config(
            session_slot_pool_size=3,
            session_slot_timeout_seconds=3.0,
            session_slot_timeout_per_token_seconds=0.0015,
            session_slot_max_timeout_seconds=60.0,
        )
        slot, fname, timeout = _build_slot_context(cfg, "wedge-scale", _body_for_tokens(20000))
        assert slot is not None
        # 3.0 + 0.0015 * 20000 = 33.0, below cap 60
        assert 32.0 <= timeout <= 34.0

    def test_adaptive_timeout_capped(self):
        _slot_owners.clear()
        cfg = _make_config(
            session_slot_pool_size=3,
            session_slot_timeout_seconds=3.0,
            session_slot_timeout_per_token_seconds=0.0015,
            session_slot_max_timeout_seconds=60.0,
        )
        slot, fname, timeout = _build_slot_context(cfg, "wedge-cap", _body_for_tokens(50000))
        assert timeout == 60.0  # 3.0 + 0.0015*50000 is capped at 60


# ------------------------------------------------------------------ #
# End-to-end: routing clamp and persistence clamp are identical
# ------------------------------------------------------------------ #

class TestClampsIdentical:
    def test_routing_and_persistence_share_the_same_clamp(self):
        """effective_per_slot_threshold IS the clamp for both routing and
        persistence — smoke check that the two 262K live configs agree."""
        for slots, expected in [(3, FAST_CLAMP), (2, CHEAP_CLAMP)]:
            assert effective_per_slot_threshold(262144, slots) == expected
