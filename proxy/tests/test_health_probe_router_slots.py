"""`/health` must never probe the local router's `GET /slots`
(LP-0MVVCXUDZ007XID0 AC1/AC2).

The router llama-server serialises ``GET /slots?model=...`` behind the busy
chat child's generation loop and converts a cancelled proxied poll into an
HTTP 500 (LP-0MVP7525P008RXMF). ``/health`` calls
``_probe_backend_reachable()`` on **every** request, and that probe used to put
``/slots?model=<current_model>`` at the head of its router probe path list — so
every health check logged a router-responder ``/slots`` 500 (1,171 of them in
the measured 48 h window) and could delay ``/health`` by up to the probe
timeout while the query sat behind a giant prefill.

Reachability is still answered by ``/models`` and ``/health`` on the router,
neither of which touches the child, and a ``/slots`` 5xx could never have
changed the verdict: the probe falls through to the next path on a 5xx, so
only ``/models``/``/health`` decided the outcome.
"""

from unittest.mock import AsyncMock

import httpx
import proxy.server as server
import pytest

ROUTER_PORT = 8080


class _RunningProc:
    """Stand-in llama-server process handle that reports 'alive'."""

    pid = 41395

    def poll(self):
        return None


class _Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _RecordingClient:
    """Minimal httpx client stand-in that records every requested URL."""

    def __init__(self, status_by_path=None, default_status=200):
        self.urls = []
        self._status_by_path = status_by_path or {}
        self._default_status = default_status

    async def get(self, url, timeout=None):
        self.urls.append(url)
        return _Response(self._status_by_path.get(_path(url), self._default_status))

    @property
    def paths(self):
        return [_path(url) for url in self.urls]


def _path(url: str) -> str:
    """Return the request path (with query) from an absolute probe URL."""
    without_scheme = url.split("://", 1)[-1]
    return "/" + without_scheme.split("/", 1)[1]


@pytest.fixture(autouse=True)
def _router_mode_state(monkeypatch):
    """Baseline router-mode server state shared by every test in this module."""
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "llama_router_mode": True,
                "llama_server_port": ROUTER_PORT,
                "tts_enabled": False,
            }
        },
    )
    monkeypatch.setattr(server, "current_model", "Qwen3")
    monkeypatch.setattr(server, "llama_process", _RunningProc())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "tts_process", None)
    monkeypatch.setattr(
        server,
        "backend_signal_counts",
        {
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        },
    )
    monkeypatch.setattr(server, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        server,
        "_backend_recovery_snapshot",
        lambda: {"in_progress": False, "recovery_started_at": None},
    )
    monkeypatch.setattr(server, "router_list_models", AsyncMock(return_value={"data": []}))
    monkeypatch.setattr(server, "_http_client", None)


async def _get_health() -> dict:
    """GET the public /health route and return its JSON body."""
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        return resp.json()


class TestProbeNeverPollsRouterSlots:
    """AC1: no router-responder /slots request, whatever the router state."""

    @pytest.mark.asyncio
    async def test_probe_never_requests_router_slots_when_reachable(self, monkeypatch):
        client = _RecordingClient()
        monkeypatch.setattr(server, "_http_client", client)

        reachable = await server._probe_backend_reachable(ROUTER_PORT)

        assert reachable is True
        assert client.paths, "probe must actually issue requests"
        assert not any(p.startswith("/slots") for p in client.paths), (
            f"reachability probe must not poll router /slots, got {client.paths}"
        )
        assert client.paths[0].startswith("/models"), (
            f"first probe path must be the child-free /models, got {client.paths}"
        )

    @pytest.mark.asyncio
    async def test_probe_never_requests_router_slots_when_unreachable(self, monkeypatch):
        client = _RecordingClient(default_status=503)
        monkeypatch.setattr(server, "_http_client", client)

        reachable = await server._probe_backend_reachable(ROUTER_PORT)

        assert reachable is False
        assert not any(p.startswith("/slots") for p in client.paths), (
            f"reachability probe must not poll router /slots, got {client.paths}"
        )
        # The fallback paths must still have been tried.
        assert any(p.startswith("/models") for p in client.paths)
        assert any(p.startswith("/health") for p in client.paths)

    @pytest.mark.asyncio
    async def test_probe_does_not_run_child_port_log_discovery(self, monkeypatch):
        """The probe must stay cheap — no child-port log scan on /health."""
        import proxy.router_helpers as router_helpers

        def _fail_discovery(*_args, **_kwargs):
            raise AssertionError(
                "reachability probe must not run child-port discovery (log scan)"
            )

        monkeypatch.setattr(
            router_helpers, "_discover_local_child_port", _fail_discovery
        )
        client = _RecordingClient()
        monkeypatch.setattr(server, "_http_client", client)

        assert await server._probe_backend_reachable(ROUTER_PORT) is True

    @pytest.mark.asyncio
    async def test_single_server_mode_still_probes_health_only(self, monkeypatch):
        monkeypatch.setattr(
            server,
            "config",
            {
                "server": {
                    "llama_router_mode": False,
                    "llama_server_port": ROUTER_PORT,
                    "tts_enabled": False,
                }
            },
        )
        client = _RecordingClient()
        monkeypatch.setattr(server, "_http_client", client)

        assert await server._probe_backend_reachable(ROUTER_PORT) is True
        assert [p.split("?")[0] for p in client.paths] == ["/health"]


class TestHealthReachabilityUnchanged:
    """AC2: /health still distinguishes a healthy backend from a dead one."""

    @pytest.mark.asyncio
    async def test_health_reports_backend_reachable_true(self, monkeypatch):
        client = _RecordingClient()
        monkeypatch.setattr(server, "_http_client", client)

        body = await _get_health()

        assert body["backend_reachable"] is True
        assert body["ready"] is True
        assert body["status"] == "healthy"
        assert not any(p.startswith("/slots") for p in client.paths), (
            f"/health must not poll router /slots, got {client.paths}"
        )

    @pytest.mark.asyncio
    async def test_health_reports_backend_reachable_false(self, monkeypatch):
        client = _RecordingClient(default_status=503)
        monkeypatch.setattr(server, "_http_client", client)

        body = await _get_health()

        assert body["backend_reachable"] is False
        assert body["ready"] is False
        assert body["status"] == "degraded"
        assert not any(p.startswith("/slots") for p in client.paths), (
            f"/health must not poll router /slots, got {client.paths}"
        )

    @pytest.mark.asyncio
    async def test_health_reports_backend_reachable_false_when_connection_refused(
        self, monkeypatch
    ):
        class _RefusingClient:
            def __init__(self):
                self.urls = []

            async def get(self, url, timeout=None):
                self.urls.append(url)
                raise httpx.ConnectError(
                    "connection refused",
                    request=httpx.Request("GET", url),
                )

        client = _RefusingClient()
        monkeypatch.setattr(server, "_http_client", client)

        body = await _get_health()

        assert body["backend_reachable"] is False
        assert body["ready"] is False
        assert not any(_path(u).startswith("/slots") for u in client.urls)
