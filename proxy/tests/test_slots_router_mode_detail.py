"""Router-mode ``/slots`` fence: live per-slot detail, never a fossil snapshot.

Slice 1 of LP-0MW1VDWY0007EMH7 (router-mode ``/slots`` observability
regression, incident 2026-11-04).

On 2026-11-04 the local model ran as a llama-server *router* with autoloaded
children. A model-less ``GET /slots`` is rejected by the router::

    400 {"error":{"code":400,"message":"model name is missing from the request"}}

so every proxy poll failed and ``/llama/local/status`` then served the
last-known per-slot detail cache **forever** (``slots_detail_stale=true``,
``slots_snapshot_source="snapshot"``, ``slots_snapshot_age_seconds=29825``).
The downtime dispatcher fails closed on ``slots_stale``, so no work was
scheduled for 8 h 17 m.

The ``?model=`` path and child-port discovery were both rebuilt after the
incident (LP-0MVU7ZZIA0092AXW, LP-0MVP7525P008RXMF, LP-0MVWBXEAI008YZ7B), but
no single test pinned the *composed* contract. This module is that fence: a
fake router that answers exactly like the real one, driven through the real
``GET /llama/local/status`` handler, with a negative control proving the
assertions can fail.

Deliberately **not** mocked here: ``_query_slots``, ``_query_slots_detail``,
the ``?model=`` URL construction, the single-flight/TTL payload cache, or the
status handler's detail vs snapshot decision. Only the HTTP transport and the
shared resolver's port choice are injected.
"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

#: The router's child port (children bind ``--port 0``; this is a stand-in).
CHILD_PORT = 41395
ROUTER_PORT = 8080

#: Exactly what the incident's llama-server router answered.
ROUTER_400_BODY = {
    "error": {
        "code": 400,
        "message": "model name is missing from the request",
        "type": "invalid_request_error",
    }
}

#: The two-slot detail list the router returns for a model-bearing query.
LIVE_SLOTS = [
    {"id": 0, "is_processing": False},
    {"id": 1, "is_processing": True},
]

#: A fossil snapshot: distinguishable from LIVE_SLOTS by slot identity so a
#: silent swap-in is detectable (rather than passing by coincidence).
FOSSIL_SLOTS = [
    {"slot_id": 7, "is_processing": True, "n_decoded": 999},
]

#: Production log shape: the live child-port signal is present
#: (``proxying request to model …``), the startup spawn line is rotated away.
PRODUCTION_SHAPE_LOG = (
    "[59947] main: model loaded\n"
    "srv  proxy_reques: proxying request to model mxbai-embed on port 59947\n"
    "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
)


class _FakeResponse:
    """Minimal httpx response: status code plus a JSON body."""

    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    async def json(self):
        return self._payload


class _RouterClient:
    """Async client that behaves like a router-mode llama-server.

    ``GET /slots`` without a ``model`` parameter is rejected with the router's
    400; with a ``model`` it returns ``slots``. Every requested URL is
    recorded so a test can assert which endpoint and parameters were used.
    """

    def __init__(self, require_model: bool = True, slots=None):
        self.require_model = require_model
        self.slots = LIVE_SLOTS if slots is None else list(slots)
        self.urls: list[str] = []

    async def get(self, url, **kwargs):
        self.urls.append(url)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        model = (params.get("model") or [None])[0]
        if self.require_model and not model:
            return _FakeResponse(400, ROUTER_400_BODY)
        return _FakeResponse(200, self.slots)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    # -- assertions helpers ------------------------------------------------
    @property
    def slot_urls(self) -> list[str]:
        return [u for u in self.urls if "/slots" in u]

    def model_param(self, url: str):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return (params.get("model") or [None])[0]

    def port_of(self, url: str) -> int | None:
        return urllib.parse.urlparse(url).port


class _HttpxShim:
    """``httpx`` proxy whose ``AsyncClient`` is a scripted router client.

    ``proxy.observability`` builds its own short-lived client for the detail
    phase (``httpx.AsyncClient(...)``); this shim redirects that construction
    without touching the real ``httpx`` module the test harness itself uses.
    Every other attribute (``Timeout``, the timeout/connection exception
    types) falls through to the real module.
    """

    def __init__(self, client):
        self.AsyncClient = lambda **kwargs: client

    def __getattr__(self, name):
        return getattr(httpx, name)


class _NoopLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def _status_config(router_mode: bool = True) -> dict:
    return {
        "server": {
            "llama_server_port": ROUTER_PORT,
            "llama_router_mode": router_mode,
        }
    }


def _reset_slots_state(fossil=None):
    """Clear the shared /slots caches so each poll is a real upstream read."""
    import proxy.observability as obs

    obs.reset_slots_poll_cache()
    obs._last_slot_counts_cache = None
    obs._last_slot_details_cache[:] = list(fossil or [])
    obs._last_slot_details_at = 0.0


async def _get_status(client, target_port=CHILD_PORT, fossil=None, config=None):
    """Issue ``GET /llama/local/status`` against a fake router-mode server.

    The shared resolver is stubbed to the documented child-port answer so the
    assertion target is the *detail vs snapshot* decision; child-port discovery
    itself is fenced separately by
    ``TestChildPortPreference::test_child_port_preferred_over_router_port``.

    Returns ``(payload, status_request_record)``. ``slots_snapshot_source`` /
    ``slots_snapshot_age_seconds`` / ``slots_upstream_calls`` are published on
    the ``status_request`` log line (LP-0MVVXGHT1001I5YX), not in the JSON
    body, so the record is returned alongside the payload.
    """
    import proxy.observability as obs
    from proxy.server import app

    from proxy import server

    # Seed the last-known per-slot detail cache when a test wants to observe
    # the snapshot-fallback regime.
    obs._last_slot_details_cache[:] = list(fossil or [])

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
                                with patch.object(
                                    server,
                                    "config",
                                    config if config is not None else _status_config(),
                                ):
                                    with patch.object(
                                        server, "_http_client", client
                                    ):
                                        with patch(
                                            "proxy.router_helpers"
                                            "._resolve_local_slots_target",
                                            lambda *a, **k: (None, target_port),
                                        ):
                                            with patch.object(
                                                obs,
                                                "httpx",
                                                _HttpxShim(client),
                                            ):
                                                async with httpx.AsyncClient(
                                                    transport=transport,
                                                    base_url="http://test",
                                                ) as ac:
                                                    resp = await ac.get(
                                                        "/llama/local/status"
                                                    )
    assert resp.status_code == 200
    return resp.json(), _last_status_request_record()


_STATUS_REQUEST_RECORD: list = []


class _CaptureStatusRequest(logging.Handler):
    """Collect ``status_request`` records so their ``extra`` fields are visible."""

    def emit(self, record):
        if record.getMessage() == "status_request":
            _STATUS_REQUEST_RECORD.append(record)


def _last_status_request_record():
    return _STATUS_REQUEST_RECORD[-1] if _STATUS_REQUEST_RECORD else None


class TestRouterModeDetailFence:
    """The proxy obtains live per-slot detail from a router-mode server."""

    @pytest.fixture(autouse=True)
    def _clean_caches(self):
        logger = logging.getLogger("llama-proxy")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        handler = _CaptureStatusRequest()
        logger.addHandler(handler)
        _STATUS_REQUEST_RECORD.clear()
        _reset_slots_state()
        yield
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        _reset_slots_state()

    @pytest.mark.asyncio
    async def test_model_bearing_query_returns_live_detail(self):
        """A router that 400s model-less /slots still yields live slot detail.

        This is the incident's exact shape: the model-less request is rejected,
        the model-bearing one succeeds, and the served payload must be the live
        detail — not the last-known snapshot.
        """
        client = _RouterClient(require_model=True)

        payload, record = await _get_status(client)

        assert [s["slot_id"] for s in payload["slots"]] == [0, 1]
        assert payload["slots_detail_stale"] is False
        assert payload["slot_counts_stale"] is False
        assert payload["slots_stale"] is False
        assert record is not None, "no status_request line was logged"
        assert record.slots_snapshot_source != "snapshot"

        # Every /slots request carried the router-required model parameter.
        slot_urls = client.slot_urls
        assert slot_urls, "the proxy issued no /slots request at all"
        assert all(client.model_param(u) for u in slot_urls), slot_urls
        assert all(client.model_param(u) == "Qwen3" for u in slot_urls), slot_urls

    @pytest.mark.asyncio
    async def test_repeated_polls_keep_live_slot_identity(self):
        """Two consecutive polls both see live slot identity (no fossil swap-in).

        A snapshot fallback would not merely be stale — it would serve slot ids
        the fixture never returned, so identity is the discriminator.
        """
        client = _RouterClient(require_model=True)

        first, _ = await _get_status(client)
        import proxy.observability as obs

        obs.reset_slots_poll_cache()
        second, _ = await _get_status(client)

        for payload in (first, second):
            assert [s["slot_id"] for s in payload["slots"]] == [0, 1]
            assert payload["slots_detail_stale"] is False
        assert len(client.slot_urls) >= 2

    @pytest.mark.asyncio
    async def test_model_less_only_server_falls_back_to_snapshot(self):
        """Negative control: a server that rejects model-bearing queries too.

        Without this control the fence could pass vacuously — it proves the
        assertions distinguish "detail obtained" from "snapshot served".
        """
        client = _RouterClient(require_model=True)
        client.require_model = False  # never accept a model parameter
        client.slots = []  # ...and answer an empty list (no usable detail)

        payload, record = await _get_status(client, fossil=FOSSIL_SLOTS)

        assert payload["slots_detail_stale"] is True
        assert record is not None, "no status_request line was logged"
        assert record.slots_snapshot_source == "snapshot"
        # The fossil snapshot is what gets served in this regime.
        assert [s["slot_id"] for s in payload["slots"]] == [7]


class TestChildPortPreference:
    """The shared resolver targets the discovered child port, not the router."""

    def _srv(self, tmp_path: Path):
        return SimpleNamespace(
            log_dir=tmp_path,
            llama_process=SimpleNamespace(pid=CHILD_PORT),
            current_model="Qwen3",
            config=_status_config(router_mode=True),
            logger=SimpleNamespace(
                warning=lambda *a, **k: None, debug=lambda *a, **k: None
            ),
        )

    def test_child_port_preferred_over_router_port(self, tmp_path):
        """A discoverable child port wins over the router port, with no fallback."""
        from proxy import metrics
        from proxy import router_helpers as rh

        (tmp_path / "llama-server.log").write_text(PRODUCTION_SHAPE_LOG)
        before = metrics.llama_slots_child_port_fallbacks_total.labels(
            reason="no_child_port_router_mode"
        )._value.get()

        with patch.object(rh, "is_model_known_unloaded", lambda *a, **k: None):
            endpoint, port = rh._resolve_local_slots_target(
                self._srv(tmp_path), model="Qwen3", router_port=ROUTER_PORT
            )

        after = metrics.llama_slots_child_port_fallbacks_total.labels(
            reason="no_child_port_router_mode"
        )._value.get()

        assert endpoint is None
        assert port == 59947 or port == CHILD_PORT, (
            "expected a discovered child port, not the router port"
        )
        assert port != ROUTER_PORT
        assert after == before, (
            "the no_child_port_router_mode fallback was taken even though a "
            "child port was discoverable"
        )
