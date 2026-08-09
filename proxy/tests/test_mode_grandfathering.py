"""Unit tests for the session→model grandfathering registry.

F1 of LP-0MSMF5LXO006YDNM (allow in-flight sessions to keep using their
models across fast↔cheap mode switches). These tests define the contract of
``proxy/proxy/grandfathering.py``: recording, persistence, expiry (next
scheduled transition / session-idle TTL / fallback grace), restricted-model
detection, and pruning. Time is injected as explicit epoch values so no clock
monkeypatching is needed.
"""

from datetime import datetime
from datetime import time as dt_time

import pytest
from proxy.grandfathering import (
    GrandfatheringRegistry,
    model_is_restricted,
    next_mode_transition,
)
from proxy.mode import ModeScheduleConfig

# A fixed, far-future epoch so the session-TTL bound never interferes with
# tests that exercise the schedule/grace bounds (and vice versa).
T0 = datetime(2026, 1, 1, 1, 0).timestamp()  # 2026-01-01 01:00 local
HOUR = 3600.0

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def models_with(*specs):
    """Build a ``models`` dict from (name, [provider-type, ...]) specs."""
    return {
        name: {
            "providers": [
                {"name": f"{name}-{idx}", "type": ptype}
                for idx, ptype in enumerate(ptypes)
            ]
        }
        for name, ptypes in specs
    }


def schedule(entries, enabled=True):
    """Build a ModeScheduleConfig from [(HH:MM, mode), ...]."""
    return ModeScheduleConfig(
        {"enabled": enabled, "entries": [{"time": t, "mode": m} for t, m in entries]}
    )


@pytest.fixture
def registry(tmp_path):
    """A registry backed by a temp state file, schedule disabled by default."""
    return GrandfatheringRegistry(
        tmp_path / "state.json",
        session_ttl=3 * HOUR,
        mode_schedule=None,
    )


def deadline_ts(dt: datetime) -> float:
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Recording (AC 1)
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_creates_binding(self, registry):
        assert registry.record("sess-1", "plan", mode="fast", now=T0)
        binding = registry.get("sess-1")
        assert binding is not None
        assert binding.model == "plan"
        assert binding.last_seen == T0
        assert binding.recorded_mode == "fast"

    def test_record_refreshes_last_seen_keeps_recorded_mode(self, registry):
        registry.record("sess-1", "plan", mode="fast", now=T0)
        assert registry.record("sess-1", "github", mode="cheap", now=T0 + 60)
        binding = registry.get("sess-1")
        assert binding.model == "github"  # updated
        assert binding.last_seen == T0 + 60  # refreshed
        assert binding.recorded_mode == "fast"  # original mode preserved

    @pytest.mark.parametrize(
        "session_id,model",
        [(None, "plan"), ("", "plan"), ("sess-1", None), ("sess-1", ""), (None, None)],
    )
    def test_anonymous_or_empty_never_recorded(self, registry, session_id, model):
        assert registry.record(session_id, model, mode="fast", now=T0) is False
        assert registry.get(session_id or "sess-1") is None
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Persistence (AC 2)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        reg = GrandfatheringRegistry(path, mode_schedule=None)
        reg.record("sess-1", "github", mode="fast", now=T0)
        reg.record("sess-2", "plan", mode="cheap", now=T0 + 120)
        reg.save()

        reloaded = GrandfatheringRegistry(path, mode_schedule=None)
        assert reloaded.get("sess-1").model == "github"
        assert reloaded.get("sess-1").last_seen == T0
        assert reloaded.get("sess-1").recorded_mode == "fast"
        assert reloaded.get("sess-2").model == "plan"
        assert reloaded.get("sess-2").recorded_mode == "cheap"

    def test_missing_state_file_starts_empty(self, tmp_path):
        reg = GrandfatheringRegistry(tmp_path / "absent.json", mode_schedule=None)
        assert reg.get("sess-1") is None
        assert reg.is_grandfathered(
            "sess-1", "cheap", {}, {}, now=T0
        ) is False

    def test_corrupt_state_file_starts_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json!!", encoding="utf-8")
        reg = GrandfatheringRegistry(path, mode_schedule=None)
        assert reg.get("sess-1") is None

        path.write_text('{"bindings": "wrong-shape"}', encoding="utf-8")
        reg2 = GrandfatheringRegistry(path, mode_schedule=None)
        assert reg2.get("sess-1") is None

    def test_empty_registry_persists_cleanly(self, tmp_path):
        path = tmp_path / "state.json"
        GrandfatheringRegistry(path, mode_schedule=None).save()
        reloaded = GrandfatheringRegistry(path, mode_schedule=None)
        assert reloaded.get("sess-1") is None
        assert reloaded.prune(now=T0) == 0


# ---------------------------------------------------------------------------
# Expiry (AC 3)
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_expires_after_session_ttl(self, registry):
        registry.record("sess-1", "plan", mode="fast", now=T0)
        assert registry.is_valid("sess-1", now=T0 + 3 * HOUR - 1)
        assert registry.is_valid("sess-1", now=T0 + 3 * HOUR) is False

    def test_expires_at_next_scheduled_transition(self, tmp_path):
        """With a long TTL, the next transition (10:00) is the deadline."""
        reg = GrandfatheringRegistry(
            tmp_path / "s.json",
            session_ttl=24 * HOUR,
            mode_schedule=schedule([("10:00", "fast")]),
        )
        reg.record("sess-1", "plan", mode="fast", now=T0)  # 01:00
        expected = deadline_ts(datetime(2026, 1, 1, 10, 0))
        assert reg.deadline("sess-1", now=T0) == expected
        assert reg.is_valid("sess-1", now=expected - 1)
        assert reg.is_valid("sess-1", now=expected) is False

    def test_earliest_of_ttl_and_transition_wins(self, tmp_path):
        """Session TTL (3h → 04:00) binds before the 10:00 transition."""
        reg = GrandfatheringRegistry(
            tmp_path / "s.json",
            session_ttl=3 * HOUR,
            mode_schedule=schedule([("10:00", "fast")]),
        )
        reg.record("sess-1", "plan", mode="fast", now=T0)
        assert reg.deadline("sess-1", now=T0) == T0 + 3 * HOUR

    def test_expires_at_fallback_grace_when_schedule_disabled(self, tmp_path):
        reg = GrandfatheringRegistry(
            tmp_path / "s.json",
            session_ttl=24 * HOUR,
            grace_seconds=1800,
            mode_schedule=None,
        )
        reg.record("sess-1", "plan", mode="fast", now=T0)
        assert reg.deadline("sess-1", now=T0) == T0 + 1800
        assert reg.is_valid("sess-1", now=T0 + 1799)
        assert reg.is_valid("sess-1", now=T0 + 1800) is False

    def test_disabled_schedule_uses_fallback_grace(self, tmp_path):
        """enabled: false is treated exactly like no schedule."""
        reg = GrandfatheringRegistry(
            tmp_path / "s.json",
            session_ttl=24 * HOUR,
            grace_seconds=3600,
            mode_schedule=schedule([("10:00", "fast")], enabled=False),
        )
        reg.record("sess-1", "plan", mode="fast", now=T0)
        assert reg.deadline("sess-1", now=T0) == T0 + 3600

    def test_fallback_grace_defaults_to_session_ttl(self, tmp_path):
        reg = GrandfatheringRegistry(
            tmp_path / "s.json", session_ttl=2 * HOUR, mode_schedule=None
        )
        reg.record("sess-1", "plan", mode="fast", now=T0)
        assert reg.deadline("sess-1", now=T0) == T0 + 2 * HOUR


class TestNextModeTransition:
    def test_next_transition_is_next_entry_after_now(self):
        sched = schedule([("00:01", "cheap"), ("10:00", "fast")])
        now = datetime(2026, 1, 1, 2, 0)
        assert next_mode_transition(sched, now) == datetime(2026, 1, 1, 10, 0)

    def test_wraps_to_first_entry_tomorrow(self):
        sched = schedule([("00:01", "cheap"), ("10:00", "fast")])
        now = datetime(2026, 1, 1, 23, 0)
        assert next_mode_transition(sched, now) == datetime(2026, 1, 2, 0, 1)


# ---------------------------------------------------------------------------
# Restricted-model detection (AC 4)
# ---------------------------------------------------------------------------


class TestRestrictedDetection:
    def test_absent_from_current_is_restricted(self):
        current = models_with(("plan", ["local"]))
        other = models_with(("plan", ["local"]), ("github", ["remote"]))
        assert model_is_restricted("github", current, other) is True

    def test_fewer_remote_providers_is_restricted(self):
        """plan/author/code lose opencode/opencode-go/deepseek tiers in cheap."""
        current = models_with(("plan", ["local"]))
        other = models_with(
            ("plan", ["local", "remote", "remote", "remote", "remote", "remote"])
        )
        assert model_is_restricted("plan", current, other) is True

    def test_identical_provider_sets_not_restricted(self):
        current = models_with(("plan", ["local", "remote"]))
        other = models_with(("plan", ["local", "remote"]))
        assert model_is_restricted("plan", current, other) is False

    def test_more_remote_in_current_not_restricted(self):
        current = models_with(("plan", ["local", "remote", "remote"]))
        other = models_with(("plan", ["local", "remote"]))
        assert model_is_restricted("plan", current, other) is False

    def test_local_only_model_present_in_both_not_restricted(self):
        current = models_with(("embed", ["local"]))
        other = models_with(("embed", ["local"]))
        assert model_is_restricted("embed", current, other) is False


# ---------------------------------------------------------------------------
# Grandfathering eligibility (parent AC 4: prior activity only)
# ---------------------------------------------------------------------------


class TestEligibility:
    FAST_MODELS = models_with(
        ("plan", ["local", "remote", "remote"]),
        ("github", ["remote"]),
    )
    CHEAP_MODELS = models_with(("plan", ["local"]))

    def test_session_recorded_before_switch_is_grandfathered(self, registry):
        registry.record("sess-1", "github", mode="fast", now=T0)
        assert registry.is_grandfathered(
            "sess-1", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0 + 60
        )

    def test_session_recorded_in_current_mode_not_grandfathered(self, registry):
        registry.record("sess-1", "github", mode="cheap", now=T0)
        assert registry.is_grandfathered(
            "sess-1", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0 + 60
        ) is False

    def test_new_session_after_switch_not_grandfathered(self, registry):
        """Parent AC 4: a session first seen after the switch gets no bypass."""
        registry.record("new-session", "github", mode="cheap", now=T0)
        assert registry.is_grandfathered(
            "new-session", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0
        ) is False

    def test_missing_binding_not_grandfathered(self, registry):
        assert registry.is_grandfathered(
            "unknown", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0
        ) is False

    def test_unrestricted_model_not_grandfathered(self, registry):
        """In fast mode nothing is restricted, so nothing is grandfathered."""
        registry.record("sess-1", "github", mode="cheap", now=T0)
        assert registry.is_grandfathered(
            "sess-1", "fast", self.FAST_MODELS, self.CHEAP_MODELS, now=T0
        ) is False

    def test_expired_binding_not_grandfathered(self, registry):
        registry.record("sess-1", "github", mode="fast", now=T0)
        assert registry.is_grandfathered(
            "sess-1", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0 + 3 * HOUR
        ) is False

    def test_disabled_registry_never_grandfathered(self, tmp_path):
        reg = GrandfatheringRegistry(
            tmp_path / "s.json", mode_schedule=None, enabled=False
        )
        reg.record("sess-1", "github", mode="fast", now=T0)
        assert reg.is_grandfathered(
            "sess-1", "cheap", self.CHEAP_MODELS, self.FAST_MODELS, now=T0 + 60
        ) is False


# ---------------------------------------------------------------------------
# Pruning (AC 5)
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_removes_only_expired(self, registry):
        registry.record("expired", "github", mode="fast", now=T0)
        registry.record("valid", "plan", mode="fast", now=T0 + 60)
        removed = registry.prune(now=T0 + 3 * HOUR + 1)
        assert removed == 1
        assert registry.get("expired") is None
        assert registry.get("valid") is not None

    def test_prune_keeps_nothing_when_all_valid(self, registry):
        registry.record("a", "plan", mode="fast", now=T0)
        assert registry.prune(now=T0 + 60) == 0
        assert registry.get("a") is not None

    def test_save_after_prune_reloads_clean(self, tmp_path):
        path = tmp_path / "state.json"
        reg = GrandfatheringRegistry(path, mode_schedule=None)
        reg.record("expired", "github", mode="fast", now=T0)
        reg.prune(now=T0 + 3 * HOUR + 1)
        reg.save()
        reloaded = GrandfatheringRegistry(path, mode_schedule=None)
        assert reloaded.get("expired") is None
        assert reloaded.prune(now=T0) == 0
