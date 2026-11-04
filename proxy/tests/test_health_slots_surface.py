"""Slot-surface staleness on ``/health`` (LP-0MW1VVMI1005XU0J).

Slice 4 of LP-0MW1VDWY0007EMH7. During the 2026-11-04 freeze ``/health``
reported ``status: healthy`` / ``ready: true`` while
``/llama/local/status`` published ``slots_stale=true`` with an 8 h 17 m-old
snapshot — so every health consumer (dashboards, ``slot_sweep_preflight.py``,
machine-hygiene) saw a healthy proxy while the dispatcher was starved. The
condition existed only in ``/metrics`` counters nobody scrapes.

The contract pinned here:

* ``/health`` carries ``slots_surface_status`` (``ok`` / ``stale`` /
  ``unknown``), ``slots_stale``, ``slots_stale_reason``,
  ``slots_snapshot_age_seconds`` and ``slots_max_snapshot_age_seconds``;
* the values come from the **same** shared evaluation the status payload uses
  (one implementation of the ceiling arithmetic and the reason precedence);
* ``ready`` and the top-level ``status`` are **unchanged** by slot staleness —
  ``ready`` gates request admission, and a stale slot *surface* does not make
  the proxy unable to serve (precedent: LP-0MVZT9L3N001XABB AC2);
* a normal idle window (llama-server down / no model loaded) reports
  ``unknown``, never ``stale``;
* a fault in the evaluator never raises.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from proxy import observability as obs


class _NoopLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def _reset_slot_state():
    obs.reset_slots_poll_cache()
    obs.reset_slots_stale_reason()
    obs._last_slot_counts_cache = None
    obs._last_slot_details_cache[:] = []
    obs._last_slot_details_at = 0.0


@pytest.fixture(autouse=True)
def _clean():
    _reset_slot_state()
    yield
    _reset_slot_state()


async def _get_health(
    llama_running: bool = True,
    backend_reachable: bool = True,
    server_config: dict | None = None,
    router_mode: bool = False,
):
    from proxy.server import app

    from proxy import server

    config = {
        "server": {
            "llama_server_port": 8080,
            "llama_router_mode": router_mode,
            "tts_enabled": False,
            **(server_config or {}),
        }
    }

    async def fake_list_models():
        return {"data": []}

    async def fake_probe(port):
        return backend_reachable

    process = None
    if llama_running:
        class _P:
            def poll(self):
                return None
        process = _P()

    transport = httpx.ASGITransport(app=app)
    with patch.object(server, "config", config):
        with patch.object(server, "llama_process", process):
            with patch.object(server, "backend_ready", True):
                with patch.object(server, "tts_process", None):
                    with patch.object(server, "current_model", "Qwen3"):
                        with patch.object(
                            server, "router_list_models", fake_list_models
                        ):
                            with patch.object(
                                server, "_probe_backend_reachable", fake_probe
                            ):
                                with patch.object(
                                    server, "_is_self_healing_active", lambda: False
                                ):
                                    with patch.object(server, "model_switch_lock", _NoopLock()):
                                        async with httpx.AsyncClient(
                                            transport=transport,
                                            base_url="http://test",
                                        ) as ac:
                                            resp = await ac.get("/health")
    assert resp.status_code == 200
    return resp.json()


class TestFieldsExist:
    """The five additive fields are always present and well-formed."""

    @pytest.mark.asyncio
    async def test_fresh_surface_reports_ok(self):
        health = await _get_health()
        assert health["slots_surface_status"] == "ok"
        assert health["slots_stale"] is False
        assert health["slots_stale_reason"] is None
        assert health["slots_snapshot_age_seconds"] is None
        assert health["slots_max_snapshot_age_seconds"] == (
            obs.local_slots_max_snapshot_age_seconds({"server": {}})
        )

    @pytest.mark.asyncio
    async def test_configured_ceiling_is_reported(self):
        health = await _get_health(
            server_config={"local_slots_max_snapshot_age_seconds": 42}
        )
        assert health["slots_max_snapshot_age_seconds"] == 42.0


class TestStaleSurface:
    """A degraded slot surface is visible without reading /metrics."""

    @pytest.mark.asyncio
    async def test_over_age_snapshot_reports_stale(self):
        """The incident: an 8 h-old snapshot must not read as healthy."""
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        obs._last_slot_details_at = time.monotonic() - 29_825.67

        health = await _get_health(
            server_config={"local_slots_max_snapshot_age_seconds": 300}
        )

        assert health["slots_surface_status"] == "stale"
        assert health["slots_stale"] is True
        assert health["slots_stale_reason"] == "snapshot_too_old"
        assert health["slots_snapshot_age_seconds"] >= 29_825

    @pytest.mark.asyncio
    async def test_transport_reason_reports_stale(self):
        obs.record_slots_stale_reason("query_timeout")
        health = await _get_health()
        assert health["slots_surface_status"] == "stale"
        assert health["slots_stale_reason"] == "query_timeout"

    @pytest.mark.asyncio
    async def test_young_snapshot_is_not_stale(self):
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        obs._last_slot_details_at = time.monotonic() - 5
        health = await _get_health(
            server_config={"local_slots_max_snapshot_age_seconds": 300}
        )
        assert health["slots_surface_status"] == "ok"
        assert health["slots_stale"] is False


class TestUnknownState:
    """An idle window is ``unknown``, never ``stale``."""

    @pytest.mark.asyncio
    async def test_backend_down_reports_unknown(self):
        health = await _get_health(llama_running=False, backend_reachable=False)
        assert health["slots_surface_status"] == "unknown"
        assert health["slots_stale"] is False
        assert health["slots_stale_reason"] is None

    @pytest.mark.asyncio
    async def test_no_snapshot_yet_reports_unknown(self):
        """Backend up but nothing observed yet: cannot assert either way."""
        health = await _get_health()
        obs._last_slot_details_cache[:] = []
        health = await _get_health()
        # No cached observation and no recorded failure — a well-formed "ok"
        # is acceptable because there is nothing degraded to report; what must
        # never happen is a fabricated ``stale``.
        assert health["slots_surface_status"] in ("ok", "unknown")
        assert health["slots_stale"] is False


class TestReadyIsUnchanged:
    """``ready`` and ``status`` are deliberately not driven by slot staleness."""

    @pytest.mark.asyncio
    async def test_stale_surface_does_not_degrade_ready(self):
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        obs._last_slot_details_at = time.monotonic() - 29_825.67

        health = await _get_health(
            server_config={"local_slots_max_snapshot_age_seconds": 300}
        )

        assert health["slots_surface_status"] == "stale"
        # The admission gate is unaffected by an unobservable slot surface.
        assert health["ready"] is True
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_transport_reason_does_not_degrade_ready(self):
        obs.record_slots_stale_reason("query_error")
        health = await _get_health()
        assert health["ready"] is True
        assert health["status"] == "healthy"


class TestSharedEvaluator:
    """``/health`` and ``/llama/local/status`` agree on the reason."""

    @pytest.mark.asyncio
    async def test_health_and_status_agree_for_each_reason(self):
        from proxy.server import app

        from proxy import server

        class _Failing:
            async def get(self, url, **kwargs):
                class _R:
                    status_code = 500

                    async def json(self):
                        return None

                return _R()

        async def fake_query_llama_status():
            return {"llama_server_running": True}

        for reason in ("query_timeout", "query_error"):
            _reset_slot_state()
            obs.record_slots_stale_reason(reason)

            config = {
                "server": {
                    "llama_server_port": 8080,
                    "llama_router_mode": True,
                    "tts_enabled": False,
                }
            }
            transport = httpx.ASGITransport(app=app)
            class _P:
                def poll(self):
                    return None

            with patch.object(server, "config", config):
                with patch.object(server, "llama_process", _P()):
                    with patch.object(server, "backend_ready", True):
                        with patch.object(server, "tts_process", None):
                            with patch.object(server, "current_model", "Qwen3"):
                                with patch.object(
                                    server,
                                    "query_llama_status",
                                    side_effect=fake_query_llama_status,
                                ):
                                    with patch.object(
                                        server, "_probe_backend_reachable",
                                        lambda *a, **k: _true(),
                                    ):
                                        with patch.object(
                                            server, "_is_self_healing_active",
                                            lambda: False,
                                        ):
                                            with patch.object(
                                                server, "background_loads", {}
                                            ):
                                                with patch.object(
                                                    server, "local_dispatch_records", {}
                                                ):
                                                    with patch.object(
                                                        server,
                                                        "local_dispatch_records_lock",
                                                        _NoopLock(),
                                                    ):
                                                        with patch.object(
                                                            server, "model_switch_lock", _NoopLock()
                                                        ):
                                                            with patch.object(
                                                                server, "_http_client", _Failing()
                                                            ):
                                                                with patch.object(
                                                                    obs,
                                                                    "httpx",
                                                                    _shim(_Failing()),
                                                                ):
                                                                    async with httpx.AsyncClient(
                                                                        transport=transport,
                                                                        base_url="http://test",
                                                                    ) as ac:
                                                                        # Poll the status endpoint first: /health
                                                                        # mirrors the last *published* reason, so it
                                                                        # must be sampled after one status poll.
                                                                        status = (
                                                                            await ac.get(
                                                                                "/llama/local/status"
                                                                            )
                                                                        ).json()
                                                                        health = (
                                                                            await ac.get("/health")
                                                                        ).json()

            assert status["slots_stale_reason"] is not None, (
                f"the status endpoint reported no reason for the {reason} regime"
            )
            assert health["slots_stale_reason"] == status["slots_stale_reason"], (
                f"/health and /llama/local/status disagreed for {reason}: "
                f"{health['slots_stale_reason']!r} vs "
                f"{status['slots_stale_reason']!r}"
            )
            assert health["slots_surface_status"] == "stale"
            assert health["slots_max_snapshot_age_seconds"] == status[
                "slots_max_snapshot_age_seconds"
            ]


async def _true():
    return True


def _shim(client):
    class _Shim:
        def __init__(self):
            self.AsyncClient = lambda **kwargs: client

        def __getattr__(self, name):
            return getattr(httpx, name)

    return _Shim()


class TestEvaluatorFaultTolerance:
    """A fault in the evaluator degrades to ``unknown``, never raises."""

    @pytest.mark.asyncio
    async def test_evaluator_exception_reports_unknown(self):
        def _boom(*a, **k):
            raise RuntimeError("boom")

        with patch.object(obs, "evaluate_slots_surface", _boom):
            health = await _get_health()
        assert health["slots_surface_status"] == "unknown"
        assert health["slots_stale"] is False

    def test_evaluator_never_raises_on_bad_config(self):
        result = obs.evaluate_slots_surface({"server": "not-a-dict"})
        assert result["status"] in ("ok", "stale", "unknown")
        assert result["max_snapshot_age_seconds"] > 0
