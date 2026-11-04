"""Bound the served slot snapshot (LP-0MW1VVKPF001ZAV5).

Slice 3 of LP-0MW1VDWY0007EMH7 — **this is the actual defect**. On 2026-11-04
``/llama/local/status`` published ``slots_snapshot_age_seconds=29825.67``
(8 h 17 m) **and kept claiming availability** from that fossil:
``available_slots=2 capacity_hint=idle``. ``slots_snapshot_age_seconds`` was
reported but never enforced, so a single frozen snapshot could starve the
dispatcher indefinitely.

The counts cache was already bounded (``SLOT_COUNTS_STALE_AFTER_SECONDS``,
1 h); the *per-slot detail* cache (``_last_slot_details_cache`` /
``_last_slot_details_at``) had no bound at all. This module pins the new
``local_slots_max_snapshot_age_seconds`` ceiling:

* the knob resolves config-first, then env, then a default derived from the
  status poll cadence (never a hard-coded magic hour);
* past the ceiling the proxy stops serving the stale list: ``slots`` is empty,
  availability is not claimed from the fossil, and the reason is
  ``snapshot_too_old``;
* below the ceiling the existing graceful-degradation behaviour is unchanged,
  including the true age;
* recovery is automatic — the next successful ``/slots?model=…`` poll restores
  a fresh surface with no restart.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from proxy import observability as obs

# ======================================================================
# The knob
# ======================================================================


class TestCeilingKnob:
    """``local_slots_max_snapshot_age_seconds`` resolution order."""

    def test_default_is_derived_from_the_poll_cadence(self):
        """The default is cadence-derived, not a magic hour."""
        default = obs.local_slots_max_snapshot_age_seconds(config={})
        assert default == max(
            120.0, 10.0 * obs.STATUS_POLL_CADENCE_SECONDS
        )
        assert default != 3600.0, (
            "the ceiling must be a small multiple of the poll cadence, not the "
            "counts-cache hour"
        )

    def test_config_value_wins(self):
        cfg = {"server": {"local_slots_max_snapshot_age_seconds": 45}}
        assert obs.local_slots_max_snapshot_age_seconds(config=cfg) == 45.0

    def test_env_value_used_when_config_absent(self, monkeypatch):
        monkeypatch.setenv("LOCAL_SLOTS_MAX_SNAPSHOT_AGE_SECONDS", "77")
        assert obs.local_slots_max_snapshot_age_seconds(config={}) == 77.0

    def test_config_beats_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_SLOTS_MAX_SNAPSHOT_AGE_SECONDS", "77")
        cfg = {"server": {"local_slots_max_snapshot_age_seconds": 45}}
        assert obs.local_slots_max_snapshot_age_seconds(config=cfg) == 45.0

    @pytest.mark.parametrize("bad", ["not-a-number", "", "0", "-5", None])
    def test_unparseable_value_falls_back_to_the_default(self, bad, monkeypatch):
        """A bad value never raises and never disables the bound."""
        monkeypatch.setenv("LOCAL_SLOTS_MAX_SNAPSHOT_AGE_SECONDS", str(bad))
        default = max(120.0, 10.0 * obs.STATUS_POLL_CADENCE_SECONDS)
        assert obs.local_slots_max_snapshot_age_seconds(config={}) == default
        cfg = {"server": {"local_slots_max_snapshot_age_seconds": bad}}
        assert obs.local_slots_max_snapshot_age_seconds(config=cfg) == default

    def test_missing_server_section_is_safe(self):
        default = max(120.0, 10.0 * obs.STATUS_POLL_CADENCE_SECONDS)
        assert obs.local_slots_max_snapshot_age_seconds(config=None) == default
        assert obs.local_slots_max_snapshot_age_seconds(config={"server": None}) == default


class TestTooOldPredicate:
    """``slots_snapshot_too_old`` is a plain, total comparison."""

    @pytest.mark.parametrize(
        "age,ceiling,expected",
        [
            (0.0, 300.0, False),
            (299.9, 300.0, False),
            (300.0, 300.0, False),  # the ceiling is the last acceptable age
            (300.1, 300.0, True),
            (29825.67, 300.0, True),
        ],
    )
    def test_comparison(self, age, ceiling, expected):
        assert obs.slots_snapshot_too_old(age, ceiling) is expected

    def test_unknown_age_is_not_too_old(self):
        """No age is no evidence of staleness — never fail on a missing age."""
        assert obs.slots_snapshot_too_old(None, 300.0) is False


# ======================================================================
# Payload enforcement
# ======================================================================


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload


class _NoopLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class _ScriptedClient:
    """Router-mode client whose ``/slots`` behaviour a test scripts per poll."""

    def __init__(self, ok=True, slots=None):
        self.ok = ok
        self.slots = (
            [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]
            if slots is None
            else list(slots)
        )

    async def get(self, url, **kwargs):
        if not self.ok:
            return _Resp(500)
        return _Resp(200, self.slots)


class _HttpxShim:
    def __init__(self, client):
        self.AsyncClient = lambda **kwargs: client

    def __getattr__(self, name):
        return getattr(httpx, name)


FOSSIL = [
    {"slot_id": 0, "is_processing": False, "n_decoded": None},
    {"slot_id": 1, "is_processing": False, "n_decoded": None},
]


async def _get_status(client, server_config=None, fossil=None, age_seconds=None):
    """Poll ``/llama/local/status`` against a scripted router-mode server.

    ``age_seconds`` back-dates the seeded fossil cache's stamp so the served
    snapshot has a known, deterministic age without patching ``time``.
    """
    from proxy import server
    from proxy.server import app

    config = {
        "server": {
            "llama_server_port": 8080,
            "llama_router_mode": True,
            **(server_config or {}),
        }
    }

    obs._last_slot_details_cache[:] = list(fossil or [])
    if fossil:
        obs._last_slot_details_at = time.monotonic() - float(age_seconds or 0.0)
    else:
        obs._last_slot_details_at = 0.0

    async def fake_query_llama_status():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)
    with patch(
        "proxy.server.query_llama_status", side_effect=fake_query_llama_status
    ):
        with patch.object(server, "current_model", "Qwen3"):
            with patch.object(server, "model_switch_refcount", 0):
                with patch.object(server, "model_switch_lock", _NoopLock()):
                    with patch.object(server, "background_loads", {}):
                        with patch.object(server, "local_dispatch_records", {}):
                            with patch.object(
                                server, "local_dispatch_records_lock", _NoopLock()
                            ):
                                with patch.object(server, "config", config):
                                    with patch.object(server, "_http_client", client):
                                        with patch(
                                            "proxy.router_helpers"
                                            "._resolve_local_slots_target",
                                            lambda *a, **k: (None, 41395),
                                        ):
                                            with patch.object(
                                                obs, "httpx", _HttpxShim(client)
                                            ):
                                                async with httpx.AsyncClient(
                                                    transport=transport,
                                                    base_url="http://test",
                                                ) as ac:
                                                    resp = await ac.get(
                                                        "/llama/local/status"
                                                    )
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture(autouse=True)
def _clean_caches():
    obs.reset_slots_poll_cache()
    obs.reset_slots_stale_reason()
    obs._last_slot_counts_cache = None
    obs._last_slot_details_cache[:] = []
    obs._last_slot_details_at = 0.0
    yield
    obs.reset_slots_poll_cache()
    obs.reset_slots_stale_reason()
    obs._last_slot_counts_cache = None
    obs._last_slot_details_cache[:] = []
    obs._last_slot_details_at = 0.0


class TestClockAdvancePastCeiling:
    """The defect: an 8-hour freeze must be impossible."""

    @pytest.mark.asyncio
    async def test_payload_stops_claiming_availability_past_the_ceiling(self):
        """The incident's own numbers: a 29825 s-old two-slot fossil."""
        payload = await _get_status(
            _ScriptedClient(ok=False),
            server_config={"local_slots_max_snapshot_age_seconds": 300},
            fossil=FOSSIL,
            age_seconds=29_825.67,
        )

        # No availability is claimed from the fossil.
        assert payload["slots"] == []
        assert payload["available_slots"] == 0
        assert payload["total_slots"] == 0
        assert payload["capacity_hint"] != "idle"

        # ...and the cause is explicit and machine-readable.
        assert payload["slots_stale"] is True
        assert payload["slots_detail_stale"] is True
        assert payload["slots_stale_reason"] == "snapshot_too_old"

    @pytest.mark.asyncio
    async def test_age_is_still_published_truthfully(self):
        """The true age is reported even when the list is withheld."""
        payload = await _get_status(
            _ScriptedClient(ok=False),
            server_config={"local_slots_max_snapshot_age_seconds": 300},
            fossil=FOSSIL,
            age_seconds=1000,
        )
        assert payload["slots_snapshot_age_seconds"] >= 1000
        assert payload["slots_snapshot_source"] == "snapshot"

    @pytest.mark.asyncio
    async def test_default_ceiling_bounds_the_incident_age(self):
        """With no knob configured, the derived default still rejects 8 h."""
        payload = await _get_status(
            _ScriptedClient(ok=False),
            fossil=FOSSIL,
            age_seconds=29_825.67,
        )
        assert payload["slots_stale_reason"] == "snapshot_too_old"
        assert payload["slots"] == []


class TestBelowTheCeiling:
    """Existing graceful degradation is unchanged below the ceiling."""

    @pytest.mark.asyncio
    async def test_stale_snapshot_still_served_below_the_ceiling(self):
        payload = await _get_status(
            _ScriptedClient(ok=False),
            server_config={"local_slots_max_snapshot_age_seconds": 300},
            fossil=FOSSIL,
            age_seconds=10,
        )
        # Slot identity survives a transient failure, as before.
        assert [s["slot_id"] for s in payload["slots"]] == [0, 1]
        assert payload["slots_snapshot_source"] == "snapshot"
        assert payload["slots_detail_stale"] is True
        assert payload["slots_stale_reason"] in ("query_error", "query_timeout")

    @pytest.mark.asyncio
    async def test_ceiling_midpoint_is_inclusive(self):
        """An age just under the ceiling is still served."""
        payload = await _get_status(
            _ScriptedClient(ok=False),
            server_config={"local_slots_max_snapshot_age_seconds": 300},
            fossil=FOSSIL,
            age_seconds=295,
        )
        assert payload["slots"] != []
        assert payload["slots_stale_reason"] != "snapshot_too_old"


class TestAutomaticRecovery:
    """No restart is required: a successful poll clears the condition."""

    @pytest.mark.asyncio
    async def test_successful_poll_after_the_ceiling_recovers(self):
        client = _ScriptedClient(ok=False)
        stale = await _get_status(
            client,
            server_config={"local_slots_max_snapshot_age_seconds": 300},
            fossil=FOSSIL,
            age_seconds=29_825.67,
        )
        assert stale["slots_stale_reason"] == "snapshot_too_old"

        # The very next successful /slots?model=... poll restores the surface.
        client.ok = True
        obs.reset_slots_poll_cache()
        fresh = await _get_status(
            client, server_config={"local_slots_max_snapshot_age_seconds": 300}
        )

        assert fresh["slots_stale"] is False
        assert fresh["slots_stale_reason"] is None
        assert [s["slot_id"] for s in fresh["slots"]] == [0, 1]
        assert fresh["available_slots"] == 2


class TestEffectiveCeilingIsPublished:
    """The applied ceiling is published so a consumer can judge severity."""

    @pytest.mark.asyncio
    async def test_payload_carries_the_applied_ceiling(self):
        payload = await _get_status(
            _ScriptedClient(ok=True),
            server_config={"local_slots_max_snapshot_age_seconds": 123},
        )
        assert payload["slots_max_snapshot_age_seconds"] == 123.0

    @pytest.mark.asyncio
    async def test_payload_carries_the_default_when_unset(self):
        payload = await _get_status(_ScriptedClient(ok=True))
        assert payload["slots_max_snapshot_age_seconds"] == (
            obs.local_slots_max_snapshot_age_seconds(config={})
        )
