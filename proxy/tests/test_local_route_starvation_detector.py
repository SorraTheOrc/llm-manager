"""Self-throttled local-route starvation warning (LP-0MVWFKIVO007EGDU).

Parent: LP-0MVWEZCB0006WPBQ (AC11).

The defect this detector closes
-------------------------------
The 2026-10-31 measurement found ``local-qwen3`` armed **217 times in 5.5 h**
(median gap 67 s ≈ the 60 s TTL) while ``/llama/local/status`` kept reporting
``available_slots: 2``, ``capacity_hint: idle`` and
``llama_server_running: true``. Local share for the window was **2.8 %** (165
local vs 5 707 remote streams). Nothing in the log said "the local route is
armed while the engine is idle" — the evidence was spread over hundreds of
individual arm lines, so the episode could only be diagnosed by archaeology.

Contract pinned here (parent AC11 and its sub-criteria)
-------------------------------------------------------
* **AC11** — at most one WARNING per 10 minutes, state-only: the tracker is fed
  from the **existing** arm sites and evaluated on those events. No timer, no
  task, no HTTP call, no new poll.
* **AC11b** — the warning fires only when the engine is healthy
  (``llama_server_running=true``, ``available_slots > 0``, not in the
  cold-start grace window) **and** the armed intervals cover a sustained
  fraction of the rolling window (default: half of 600 s).
* **AC11c** — the line names per-reason counts in canonical routing tokens
  (``local_provider_cooldown``, ``measured_cache_deficit``,
  ``prefill_thrash_cooldown``).
* **AC11d** — a single genuine fault (one arm) never fires it, even when that
  one arm spans the whole window.
* **AC11e** — everything below runs on an injected clock and injected engine
  health: deterministic, no sleeps.

Every test asserts observable behaviour — the WARNING line the operator reads,
or the read-only tracker snapshot — never the module's internals.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import proxy.local_route_starvation as detector
import pytest

from proxy import cold_start, provider, router_helpers

#: Engine state that satisfies parent AC11b (healthy and idle).
HEALTHY = {
    "llama_server_running": True,
    "available_slots": 2,
    "cold_start": False,
}

#: Engine stopped: a route arm is expected — not starvation.
UNHEALTHY_STOPPED = {
    "llama_server_running": False,
    "available_slots": 2,
    "cold_start": False,
}

#: Engine busy (no free slot): a route arm is explained — not starvation.
UNHEALTHY_BUSY = {
    "llama_server_running": True,
    "available_slots": 0,
    "cold_start": False,
}

#: Slot counts unknown (no fresh ``/slots`` sample): not assertable, so silent.
UNKNOWN_SLOTS = {
    "llama_server_running": True,
    "available_slots": None,
    "cold_start": False,
}

#: Warm-up deferral is expected behaviour, never starvation.
COLD_START = {
    "llama_server_running": True,
    "available_slots": 2,
    "cold_start": True,
}

WARNING_PREFIX = detector.WARNING_TOKEN


def _warnings(caplog) -> list[logging.LogRecord]:
    """The detector's WARNING records from *caplog*."""
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and record.getMessage().startswith(WARNING_PREFIX)
    ]


def _make_http_response(status_code: int) -> MagicMock:
    """Minimal response stand-in for the provider error handlers."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return response


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the tracker and the shared provider/thrash state per test."""
    detector.reset_local_route_starvation_tracker()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    router_helpers._prefill_thrash_cooldowns.clear()
    router_helpers._prefill_thrash_cancel_tokens.clear()
    cold_start.mark_warm()
    yield
    detector.reset_local_route_starvation_tracker()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    router_helpers._prefill_thrash_cooldowns.clear()
    router_helpers._prefill_thrash_cancel_tokens.clear()
    cold_start.mark_warm()


class TestSustainedArmedWindow:
    """AC11/AC11c: a sustained armed window warns once and names the reasons."""

    def test_sustained_window_fires_once_with_reason_counts(self, caplog):
        """Six contiguous 60 s arms reach half the 600 s window → one line."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        reasons = [
            "local_provider_cooldown",
            "local_provider_cooldown",
            "local_provider_cooldown",
            "measured_cache_deficit",
            "measured_cache_deficit",
            "prefill_thrash_cooldown",
            "local_provider_cooldown",
        ]
        fired = [
            detector.record_local_route_arm(
                reason, 60.0, now=1000.0 + 60.0 * index, health=HEALTHY
            )
            for index, reason in enumerate(reasons)
        ]

        # An arm contributes coverage only up to the instant it is evaluated,
        # so the armed intervals cover 300 s of the 600 s window on the sixth
        # arm; the seventh is inside the 10-minute throttle and stays silent.
        assert fired == [False, False, False, False, False, True, False]
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "armed_s=300" in message
        assert "window_s=600" in message
        assert "armed_fraction=0.50" in message
        assert "arms=6" in message
        assert (
            "reasons=local_provider_cooldown:3,measured_cache_deficit:2,"
            "prefill_thrash_cooldown:1"
        ) in message
        # The healthy-engine evidence rides on the same line, so the operator
        # can tell "route armed while idle" from "route armed because it broke".
        assert "available_slots=2" in message
        assert "llama_server_running=true" in message

    def test_warning_is_suppressed_inside_the_ten_minute_throttle(self, caplog):
        """A sustained window warns once, then again only after 10 minutes."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        fired = [
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1000.0 + 60.0 * index,
                health=HEALTHY,
            )
            for index in range(6)
        ]
        assert fired[-1] is True
        assert len(_warnings(caplog)) == 1

        # Keep the route armed every minute for the next nine minutes: the
        # state is still starved but the throttle holds the line back.
        for index in range(6, 15):
            assert (
                detector.record_local_route_arm(
                    "local_provider_cooldown",
                    60.0,
                    now=1000.0 + 60.0 * index,
                    health=HEALTHY,
                )
                is False
            )
        assert len(_warnings(caplog)) == 1

        # Exactly one throttle window after the first warning (t=1300): warn.
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown", 60.0, now=1900.0, health=HEALTHY
            )
            is True
        )
        assert len(_warnings(caplog)) == 2

    def test_overlapping_arms_are_counted_once(self):
        """Concurrent sessions refused by one entry cooldown are one period."""
        for _ in range(3):
            detector.record_local_route_arm(
                "local_provider_cooldown", 60.0, now=1000.0, health=HEALTHY
            )
        snapshot = detector.starvation_snapshot(now=1060.0, health=HEALTHY)
        assert snapshot["arms"] == 3
        assert snapshot["armed_seconds"] == 60.0
        assert snapshot["armed_fraction"] == 0.1

    def test_health_is_evaluated_at_each_arm(self, caplog):
        """Arms recorded while the engine is busy stay silent until it frees."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        for index in range(5):
            assert (
                detector.record_local_route_arm(
                    "local_provider_cooldown",
                    60.0,
                    now=1000.0 + 60.0 * index,
                    health=UNHEALTHY_BUSY,
                )
                is False
            )
        assert _warnings(caplog) == []

        # The engine frees a slot; the window is still armed, so this arm is
        # the first one that can honestly say "idle engine, armed route".
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown", 60.0, now=1300.0, health=HEALTHY
            )
            is True
        )
        assert len(_warnings(caplog)) == 1


class TestNoFalsePositive:
    """AC11b/AC11d: healthy-engine gate and the single-arm exemption."""

    def test_single_arm_never_fires(self, caplog):
        """One genuine fault is normal: even a window-spanning arm stays silent."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown", 600.0, now=1000.0, health=HEALTHY
            )
            is False
        )
        assert _warnings(caplog) == []

    def test_scattered_short_arms_stay_below_the_fraction(self, caplog):
        """Two 60 s arms a minute apart cover 120 s of 600 s → silent."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        fired = [
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1000.0 + 100.0 * index,
                health=HEALTHY,
            )
            for index in range(2)
        ]
        assert fired == [False, False]
        assert _warnings(caplog) == []

    @pytest.mark.parametrize(
        "health",
        [UNHEALTHY_STOPPED, UNHEALTHY_BUSY, UNKNOWN_SLOTS, COLD_START],
        ids=["engine-stopped", "no-free-slot", "slots-unknown", "cold-start"],
    )
    def test_unhealthy_engine_never_fires(self, health, caplog):
        """Every non-healthy engine state suppresses the warning."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        fired = [
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1000.0 + 60.0 * index,
                health=health,
            )
            for index in range(6)
        ]
        assert not any(fired)
        assert _warnings(caplog) == []

    def test_default_health_probe_reports_unknown_engine_silently(self, caplog):
        """Without a live llama-server the probe reports healthy=False."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        fired = [
            detector.record_local_route_arm(
                "local_provider_cooldown", 60.0, now=1000.0 + 60.0 * index
            )
            for index in range(6)
        ]
        assert not any(fired)
        assert _warnings(caplog) == []
        snapshot = detector.starvation_snapshot(now=1300.0)
        assert snapshot["engine_healthy"] is False


class TestConfiguration:
    """The knobs are value-change configurable and malformed values fail safe."""

    def test_disabled_by_config_value(self, caplog):
        """``local_route_starvation_enabled: 0`` is the rollback."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        config = {"server": {"local_route_starvation_enabled": 0}}
        fired = [
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1000.0 + 60.0 * index,
                config=config,
                health=HEALTHY,
            )
            for index in range(6)
        ]
        assert not any(fired)
        assert _warnings(caplog) == []
        assert detector.starvation_snapshot(now=1000.0, config=config)["arms"] == 0

    def test_window_and_threshold_are_configurable(self, caplog):
        """A 120 s window with a 0.5 fraction fires on two contiguous arms."""
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        config = {
            "server": {
                "local_route_starvation_window_seconds": 120,
                "local_route_starvation_min_armed_fraction": 0.5,
            }
        }
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1000.0,
                config=config,
                health=HEALTHY,
            )
            is False
        )
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown",
                60.0,
                now=1060.0,
                config=config,
                health=HEALTHY,
            )
            is True
        )
        assert len(_warnings(caplog)) == 1

    def test_min_arms_knob_can_never_be_zero(self):
        """A malformed ``min_arms`` falls back to the default, never to 0."""
        settings = detector.starved_warning_settings(
            {"server": {"local_route_starvation_min_arms": 0}}
        )
        assert settings["min_arms"] == detector.DEFAULT_MIN_ARMS

    def test_malformed_window_falls_back_to_the_default(self):
        """A config typo cannot disable the detector."""
        settings = detector.starved_warning_settings(
            {"server": {"local_route_starvation_window_seconds": "ten-minutes"}}
        )
        assert settings["window_seconds"] == detector.DEFAULT_WINDOW_SECONDS

    def test_zero_duration_or_blank_reason_records_nothing(self):
        """A disabled cooldown (0 s) or an unnamed arm is not an armed route."""
        assert (
            detector.record_local_route_arm(
                "local_provider_cooldown", 0.0, now=1000.0, health=HEALTHY
            )
            is False
        )
        assert (
            detector.record_local_route_arm("", 60.0, now=1000.0, health=HEALTHY)
            is False
        )
        assert detector.starvation_snapshot(now=1000.0, health=HEALTHY)["arms"] == 0


class TestSnapshot:
    """AC11's read-only state surface (no new status fields, no routing input)."""

    def test_snapshot_is_stable_and_reads_the_window(self):
        detector.record_local_route_arm(
            "local_provider_cooldown", 60.0, now=1000.0, health=HEALTHY
        )
        detector.record_local_route_arm(
            "measured_cache_deficit", 120.0, now=1060.0, health=HEALTHY
        )
        first = detector.starvation_snapshot(now=1120.0, health=HEALTHY)
        second = detector.starvation_snapshot(now=1120.0, health=HEALTHY)

        assert first == second
        assert first["arms"] == 2
        # The second arm starts exactly where the first one ends, so the union
        # is 1000→1120 = 120 s (not 180 s): overlapping arms never double-count.
        assert first["armed_seconds"] == 120.0
        assert first["armed_fraction"] == 0.2
        assert first["reasons"] == {
            "local_provider_cooldown": 1,
            "measured_cache_deficit": 1,
        }
        assert first["last_warning_age_s"] is None
        assert first["engine_healthy"] is True
        assert first["engine"] == {
            "llama_server_running": True,
            "available_slots": 2,
            "cold_start": False,
        }

    def test_snapshot_ages_out_events_past_the_window(self):
        detector.record_local_route_arm(
            "local_provider_cooldown", 60.0, now=1000.0, health=HEALTHY
        )
        snapshot = detector.starvation_snapshot(now=1700.0, health=HEALTHY)
        assert snapshot["arms"] == 0
        assert snapshot["armed_seconds"] == 0.0


class TestArmSiteWiring:
    """The tracker is fed by the existing arm sites, not by a new poll."""

    def test_reason_tokens_are_the_canonical_routing_vocabulary(self):
        """AC11c: the counted tokens are the ones the rest of the proxy uses.

        The module spells each token as a literal (so ``ruff --fix`` cannot
        strip a re-exported import out of the import graph), which is only safe
        while the literals match their owning definitions and the documented
        ``LOCAL_FALLBACK_REASON_TOKENS`` vocabulary.
        """
        from proxy.measured_cache_abort import MEASURED_CACHE_DEFICIT_REASON

        from proxy import admission_reserve, observability

        assert (
            detector.LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN
            == observability.LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN
        )
        assert (
            detector.LOCAL_ROUTE_REASON_MEASURED_CACHE_DEFICIT
            == MEASURED_CACHE_DEFICIT_REASON
        )
        assert detector.LOCAL_ROUTE_REASON_MEASURED_CACHE_DEFICIT in (
            admission_reserve.LOCAL_FALLBACK_REASON_TOKENS
        )
        assert detector.LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN in (
            admission_reserve.LOCAL_FALLBACK_REASON_TOKENS
        )
        assert detector.LOCAL_ROUTE_REASON_PREFILL_THRASH in (
            provider.REAL_EXPIRY_FALLBACK_REASONS
        )

    def test_local_http_error_arms_the_tracker(self, monkeypatch):
        """A quarantined local entry records ``local_provider_cooldown``."""
        monkeypatch.setattr(provider, "_persist_provider_state_best_effort", lambda: None)

        attempts: list[dict] = []
        provider._handle_http_error_with_cooldown(
            _make_http_response(500),
            "local-qwen3",
            "local",
            cooldown_seconds=60.0,
            attempts=attempts,
            body_text="internal error",
            config={"server": {"provider_cooldown_seconds": 60}},
        )

        snapshot = detector.starvation_snapshot(health=HEALTHY)
        assert snapshot["arms"] == 1
        assert snapshot["reasons"] == {"local_provider_cooldown": 1}
        # The arm is observed, not invented: the entry really is quarantined.
        assert provider._is_provider_unavailable("local-qwen3")

    def test_remote_http_error_does_not_arm_the_local_tracker(self, monkeypatch):
        """A remote quarantine is not local-route starvation."""
        monkeypatch.setattr(provider, "_persist_provider_state_best_effort", lambda: None)

        provider._handle_http_error_with_cooldown(
            _make_http_response(500),
            "opencode-go-2",
            "remote",
            cooldown_seconds=60.0,
            attempts=[],
            body_text="internal error",
            config={"server": {"provider_cooldown_seconds": 60}},
        )

        assert detector.starvation_snapshot(health=HEALTHY)["arms"] == 0

    def test_local_connection_error_arms_the_tracker(self, monkeypatch):
        """A connection failure quarantines and records the local arm."""
        import httpx

        monkeypatch.setattr(provider, "_persist_provider_state_best_effort", lambda: None)

        attempts: list[dict] = []
        handled = provider._handle_connection_error_in_fallback(
            httpx.ConnectError("refused"),
            "local-qwen3",
            "local",
            60.0,
            attempts,
        )

        assert handled is True
        snapshot = detector.starvation_snapshot(health=HEALTHY)
        assert snapshot["reasons"] == {"local_provider_cooldown": 1}

    def test_prefill_thrash_arm_records_its_reason(self):
        """The thrash window reports the cause it was armed for."""
        config = {"server": {"local_prefill_thrash_cooldown_seconds": 120}}
        router_helpers._record_prefill_thrash_cooldown(
            "session-a",
            config,
            now=1000.0,
            reason="measured_cache_deficit",
        )
        router_helpers._record_prefill_thrash_cooldown(
            "session-b",
            config,
            now=1000.0,
            reason="prefill_thrash_cooldown",
        )
        snapshot = detector.starvation_snapshot(now=1000.0, health=HEALTHY)
        assert snapshot["arms"] == 2
        assert snapshot["reasons"] == {
            "measured_cache_deficit": 1,
            "prefill_thrash_cooldown": 1,
        }

    def test_prefill_thrash_arm_without_reason_records_nothing(self):
        """Existing callers that pass no reason keep their behaviour exactly."""
        config = {"server": {"local_prefill_thrash_cooldown_seconds": 120}}
        router_helpers._record_prefill_thrash_cooldown(
            "session-a", config, now=1000.0
        )
        assert detector.starvation_snapshot(now=1000.0, health=HEALTHY)["arms"] == 0

    def test_disabled_cooldown_records_nothing(self):
        """``local_prefill_thrash_cooldown_seconds: 0`` arms no window at all."""
        config = {"server": {"local_prefill_thrash_cooldown_seconds": 0}}
        deadline = router_helpers._record_prefill_thrash_cooldown(
            "session-a", config, now=1000.0, reason="measured_cache_deficit"
        )
        assert deadline == 0.0
        assert detector.starvation_snapshot(now=1000.0, health=HEALTHY)["arms"] == 0


class TestEndToEndThroughArmSites:
    """AC11e: real arm sites, injected clock, one bounded warning."""

    def test_real_arm_sites_emit_one_bounded_warning(self, monkeypatch, caplog):
        caplog.set_level(
            logging.WARNING, logger="llama-proxy.local_route_starvation"
        )
        monkeypatch.setattr(
            detector, "default_engine_health", lambda config=None: dict(HEALTHY)
        )
        config = {"server": {"local_prefill_thrash_cooldown_seconds": 60}}

        # Seven contiguous one-minute thrash arms, driven through the real arm
        # site with an injected clock — no sleeps, no monkeypatched time.
        for index in range(7):
            router_helpers._record_prefill_thrash_cooldown(
                f"session-{index}",
                config,
                now=5000.0 + 60.0 * index,
                reason="measured_cache_deficit",
            )

        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "arms=6" in message
        assert "reasons=measured_cache_deficit:6" in message
        assert "available_slots=2" in message
        assert "llama_server_running=true" in message
