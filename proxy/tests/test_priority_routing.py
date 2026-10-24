"""Priority-routing suite (LP-0MVDT94F0003VHUA).

Covers the acceptance criteria for the priority-based routing feature that
replaces the fast/cheap operating mode:

- AC1: header parsing (``X-Priority: low|medium|high``, absent/invalid →
  medium) and effective-priority computation incl. cheap-window low-capping
  (header unmodified, no 503/Retry-After deferral, no restart).
- AC2: per-provider ``priorities`` labels select the chain per priority;
  backward-compatible default when no provider is labelled
  (LP-0MVGPZWY0001JD1D).
- AC3: priority gates the routing knobs (queue entry, economic bypass,
  local-vs-remote preference).
- AC4: mode machinery removed (module gone, overlays gone, slot pool fixed
  at 2, no restart API).
- AC6: observability — ``X-Priority-Applied`` response header and effective-
  priority log/decision points.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from proxy.priority import (
    DEFAULT_PRIORITY,
    PRIORITY_APPLIED_HEADER,
    PRIORITY_HEADER,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    cheap_window_config,
    effective_priority,
    in_cheap_window,
    parse_priority,
    priority_from_headers,
    priority_routing_profile,
    provider_chain_for_priority,
    request_priority,
    set_request_priority,
)

from proxy import priority as prio

# ---------------------------------------------------------------------------
# AC1 — header parsing & effective priority
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    def test_absent_defaults_to_medium(self):
        assert parse_priority(None) == PRIORITY_MEDIUM
        assert parse_priority("") == PRIORITY_MEDIUM

    def test_valid_values(self):
        assert parse_priority("low") == PRIORITY_LOW
        assert parse_priority("medium") == PRIORITY_MEDIUM
        assert parse_priority("high") == PRIORITY_HIGH

    def test_case_and_whitespace_tolerated(self):
        assert parse_priority("  HIGH  ") == PRIORITY_HIGH
        assert parse_priority("Low") == PRIORITY_LOW

    def test_invalid_defaults_to_medium(self):
        assert parse_priority("urgent") == PRIORITY_MEDIUM
        assert parse_priority("!!") == PRIORITY_MEDIUM

    def test_header_lookup_case_insensitive(self):
        # Pinned to a time outside the cheap window so the header lookup is
        # deterministic regardless of the wall clock (inside 01:00–10:00 the
        # effective priority would be capped to low).
        headers = {"x-priority": "high"}
        assert (
            priority_from_headers(headers, now=datetime(2026, 1, 1, 12, 0))
            == PRIORITY_HIGH
        )


class TestCheapWindow:
    CONF = {"priority_window": {"enabled": True, "start": "01:00", "end": "10:00"}}

    def test_within_window(self):
        assert in_cheap_window(self.CONF, now=datetime(2026, 1, 1, 2, 0)) is True

    def test_before_window(self):
        assert in_cheap_window(self.CONF, now=datetime(2026, 1, 1, 0, 30)) is False

    def test_at_end_exclusive(self):
        assert in_cheap_window(self.CONF, now=datetime(2026, 1, 1, 10, 0)) is False

    def test_disabled_window_no_cap(self):
        conf = {"priority_window": {"enabled": False}}
        assert in_cheap_window(conf, now=datetime(2026, 1, 1, 3, 0)) is False

    def test_missing_config_defaults_to_schedule(self):
        cfg = cheap_window_config({})
        assert cfg["enabled"] is True
        assert (cfg["start"].hour, cfg["end"].hour) == (1, 10)

    def test_overnight_window_wraps(self):
        conf = {"priority_window": {"enabled": True, "start": "22:00", "end": "06:00"}}
        assert in_cheap_window(conf, now=datetime(2026, 1, 1, 23, 0)) is True
        assert in_cheap_window(conf, now=datetime(2026, 1, 1, 3, 0)) is True
        assert in_cheap_window(conf, now=datetime(2026, 1, 1, 12, 0)) is False

    def test_invalid_window_never_crashes(self):
        conf = {"priority_window": {"enabled": True, "start": "not-a-time"}}
        # Falls back to defaults; never raises.
        cfg = cheap_window_config(conf)
        assert cfg["start"] is not None


class TestEffectivePriority:
    CONF = {"priority_window": {"enabled": True, "start": "01:00", "end": "10:00"}}

    def test_high_capped_to_low_in_cheap_window(self):
        # AC1: a spoofed high during cheap hours routes as low — without
        # changing the header and without any 503/Retry-After deferral.
        assert (
            effective_priority("high", self.CONF, now=datetime(2026, 1, 1, 3, 0))
            == PRIORITY_LOW
        )

    def test_high_kept_outside_cheap_window(self):
        assert (
            effective_priority("high", self.CONF, now=datetime(2026, 1, 1, 12, 0))
            == PRIORITY_HIGH
        )

    def test_low_capped_stays_low(self):
        assert (
            effective_priority("low", self.CONF, now=datetime(2026, 1, 1, 3, 0))
            == PRIORITY_LOW
        )

    def test_absent_defaults_medium_outside_window(self):
        assert (
            effective_priority(None, self.CONF, now=datetime(2026, 1, 1, 12, 0))
            == PRIORITY_MEDIUM
        )

    def test_absent_is_low_inside_window(self):
        # AC1: every request routes as low during the cheap window, whatever
        # the header (even an absent one).
        assert (
            effective_priority(None, self.CONF, now=datetime(2026, 1, 1, 3, 0))
            == PRIORITY_LOW
        )


# ---------------------------------------------------------------------------
# AC2 — per-provider priority labels select the chain
# ---------------------------------------------------------------------------


def _model(providers):
    return {"providers": providers}


def _prov(name, priorities=None, ptype="remote"):
    entry = {"name": name, "type": ptype}
    if priorities is not None:
        entry["priorities"] = priorities
    return entry


class TestProviderChains:
    def test_backward_compatible_default_returns_full_chain(self):
        # No provider declares ``priorities`` → every provider serves every
        # priority (the documented legacy default for ``embed``/``github``/
        # ``private``).
        cfg = _model([_prov("a"), _prov("b")])
        assert provider_chain_for_priority(cfg, "high") == cfg["providers"]
        assert provider_chain_for_priority(cfg, "low") == cfg["providers"]
        # Medium = today's semantics (full chain).
        assert provider_chain_for_priority(cfg, "medium") == cfg["providers"]

    def test_labels_filter_the_provider_list(self):
        cfg = _model(
            [
                _prov("local", ["low", "medium", "high"], "local"),
                _prov("free", ["low", "medium", "high"]),
                _prov("paid", ["medium", "high"]),
            ]
        )
        for priority in ("low", "medium", "high"):
            names = [p["name"] for p in provider_chain_for_priority(cfg, priority)]
            expected = ["local", "free"] if priority == "low" else ["local", "free", "paid"]
            assert names == expected, priority

    def test_chain_entries_are_the_provider_dicts(self):
        cfg = _model([_prov("local", ["low"], "local"), _prov("paid", ["high"])])
        assert provider_chain_for_priority(cfg, "low") == [cfg["providers"][0]]
        assert provider_chain_for_priority(cfg, "high") == [cfg["providers"][1]]

    def test_absent_priority_is_not_widened(self):
        # Labelled model with no provider for the priority: an empty chain
        # (the config validator rejects that shape at load; routing must never
        # fall back to the full provider list).
        cfg = _model([_prov("a", ["low"]), _prov("b", ["low"])])
        assert provider_chain_for_priority(cfg, "high") == []


# ---------------------------------------------------------------------------
# AC3 — priority gates the routing knobs
# ---------------------------------------------------------------------------


class TestRoutingProfile:
    def test_high_never_queues_and_routes_remote(self):
        prof = priority_routing_profile("high", {})
        assert prof["queue"] is False
        assert prof["economic_bypass_serves_local"] is False

    def test_low_queues_and_recovers_economic(self):
        prof = priority_routing_profile("low", {})
        assert prof["queue"] is True
        assert prof["economic_bypass_serves_local"] is True

    def test_medium_is_config_driven(self):
        server = {
            "contention_queue_policy": "queue",
            "local_large_context_economic_bypass_serves_local": True,
        }
        prof = priority_routing_profile("medium", server)
        assert prof["queue"] is True
        assert prof["economic_bypass_serves_local"] is True

    def test_medium_defaults_to_fallback_when_config_absent(self):
        prof = priority_routing_profile("medium", {})
        assert prof["queue"] is False
        assert prof["economic_bypass_serves_local"] is False


class TestContentionQueuePriority:
    CONF = {"server": {"contention_queue_policy": "queue"}}

    def test_high_never_engages_queue(self):
        from proxy.provider import _contention_queue_enabled

        assert _contention_queue_enabled(self.CONF, "high") is False

    def test_low_engages_queue(self):
        from proxy.provider import _contention_queue_enabled

        assert _contention_queue_enabled(self.CONF, "low") is True

    def test_medium_engages_when_policy_queue(self):
        from proxy.provider import _contention_queue_enabled

        assert _contention_queue_enabled(self.CONF, "medium") is True

    def test_medium_fallback_policy_disengages(self):
        from proxy.provider import _contention_queue_enabled

        conf = {"server": {"contention_queue_policy": "fallback"}}
        assert _contention_queue_enabled(conf, "medium") is False


class TestEconomicBypassPriority:
    def test_low_recovers_economically(self):
        from proxy.provider import _economic_bypass_serves_local

        assert _economic_bypass_serves_local({}, "low") is True

    def test_high_routes_remote(self):
        from proxy.provider import _economic_bypass_serves_local

        assert _economic_bypass_serves_local({}, "high") is False

    def test_medium_uses_config_flag(self):
        from proxy.provider import _economic_bypass_serves_local

        assert _economic_bypass_serves_local({}, "medium") is False
        conf = {"server": {"local_large_context_economic_bypass_serves_local": True}}
        assert _economic_bypass_serves_local(conf, "medium") is True


# ---------------------------------------------------------------------------
# Request stamping / observability (AC1 decision points, AC6 header)
# ---------------------------------------------------------------------------


class _DummyRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = type("S", (), {})()


class TestRequestPriority:
    def test_stamp_and_read(self):
        req = _DummyRequest({"X-Priority": "high"})
        set_request_priority(req, "low")
        assert request_priority(req) == "low"

    def test_recompute_when_unstamped(self):
        # Cheap window disabled so the recompute path is deterministic at any
        # local time (inside 01:00–10:00 every request would cap to low).
        req = _DummyRequest({"X-Priority": "high"})
        assert (
            request_priority(req, {"priority_window": {"enabled": False}})
            == "high"
        )

    def test_default_medium(self):
        assert (
            request_priority(
                _DummyRequest(), {"priority_window": {"enabled": False}}
            )
            == "medium"
        )


class TestAppliedHeader:
    def test_applied_header_name(self):
        assert PRIORITY_APPLIED_HEADER == "X-Priority-Applied"

    def test_attached_by_top_level_endpoint(self):
        # AC6: _attach_priority_applied_header writes the effective priority
        # onto any response with mutable headers (streaming responses
        # included — headers are serialised when the response starts).
        from proxy import ui as ui_mod

        response = Response(status_code=200)
        request = _DummyRequest({"X-Priority": "high"})
        set_request_priority(request, "high")
        srv = MagicMock()
        srv.config = {"server": {}}

        ui_mod._attach_priority_applied_header(response, request, srv)
        assert response.headers[PRIORITY_APPLIED_HEADER] == "high"

    def test_cheap_window_cap_visible_in_header(self):
        # The header records the effective (capped) priority, not the client
        # claim: a high header inside the cheap window yields low.
        from proxy import ui as ui_mod

        response = Response(status_code=200)
        request = _DummyRequest({"X-Priority": "high"})
        set_request_priority(request, "low")
        srv = MagicMock()
        srv.config = {"server": {}}

        ui_mod._attach_priority_applied_header(response, request, srv)
        assert response.headers[PRIORITY_APPLIED_HEADER] == "low"


# ---------------------------------------------------------------------------
# AC4 — mode machinery removed
# ---------------------------------------------------------------------------


class TestModeRemoval:
    def test_mode_module_gone(self):
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module("proxy.mode")

    def test_grandfathering_module_gone(self):
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module("proxy.grandfathering")

    def test_config_overlays_gone(self):
        from proxy.priority import proxy_dir

        assert not (proxy_dir() / "config-fast.yaml").exists()
        assert not (proxy_dir() / "config-cheap.yaml").exists()
        assert not (proxy_dir() / "mode_schedule.yaml").exists()

    def test_slot_pool_fixed_at_2(self):
        from proxy.utils import load_config

        cfg = load_config()
        assert cfg["server"]["session_slot_pool_size"] == 2

    def test_priority_window_declared(self):
        from proxy.utils import load_config

        cfg = load_config()
        window = cfg["server"]["priority_window"]
        assert window["enabled"] is True
        assert window["start"] == "01:00"
        assert window["end"] == "10:00"

    def test_models_declare_priorities_on_every_provider(self):
        from proxy.utils import load_config

        cfg = load_config()
        for name in ("plan", "author", "code", "compact"):
            providers = cfg["models"][name]["providers"]
            assert providers, name
            for entry in providers:
                assert isinstance(entry.get("priorities"), list), (
                    f"{name}/{entry.get('name')} must declare priorities"
                )

    def test_no_restart_on_policy_change_endpoint_gone(self):
        # AC4: POST /admin/set-mode / GET /admin/mode are removed from the
        # router, so no client can trigger a routing-policy restart.
        from proxy.server import app

        paths = {r.path for r in app.routes}
        assert "/admin/set-mode" not in paths
        assert "/admin/mode" not in paths
        assert "/admin/mode-switch-lock" not in paths


# ---------------------------------------------------------------------------
# AC1/AC3 log decision points
# ---------------------------------------------------------------------------


class TestDecisionPointLogging:
    def test_fallback_cycle_logs_priority(self):
        from proxy import provider as prov

        request = _DummyRequest({"X-Priority": "high"})
        # effective_chain_for_request stamps the priority on the request so
        # decision points (chain selection, queue entry) log `priority=` and
        # the response header can be attached.
        # Cheap window disabled so the stamp reflects the declared header
        # priority at any local time (inside 01:00–10:00 it would be low).
        chain = prov.effective_chain_for_request(
            request,
            {"providers": [{"name": "p", "type": "remote"}]},
            {"priority_window": {"enabled": False}},
        )
        assert chain
        assert request.state.priority == "high"


class TestCompactionGatePriority:
    def test_gate_carries_priority_not_mode(self):
        from proxy.provider import _build_compaction_gate_response

        resp = _build_compaction_gate_response(50000, 44000, "low", None, "plan")
        assert resp.status_code == 429
        assert resp.headers["X-Compaction-Gate"] == "true"
        assert resp.headers["X-Compaction-Priority"] == "low"
        assert "X-Compaction-Mode" not in resp.headers
        body = json.loads(resp.body)
        assert body["priority"] == "low"
        assert "mode" not in body
