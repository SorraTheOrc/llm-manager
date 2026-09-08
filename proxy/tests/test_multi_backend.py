
# <!-- REFACTOR-LP-0MTT1054U001OMWR
# smell: unused_import
# severity: critical
# description: Redefinition of unused `body` from line 286: `body` redefined here
# -->
"""Tests for multi-backend support (LP-0MRPILSMW004T4H8).

Coverage (Acceptance Criteria mapping):

- AC1  — ``type: local`` providers declare their own ``endpoint`` URL;
         omitting it falls back to ``http://localhost:{llama_server_port}``.
- AC2  — dispatch lease records are keyed per-endpoint
         ``(endpoint, session_id)``; the same session on different servers
         holds independent leases.
- AC3  — slot save directories are derived per-endpoint under the global
         ``session_slot_save_path`` root as ``{host}-{port}/`` subdirectories.
- AC4  — the fallback chain iterates multiple ``type: local`` providers and
         falls through when the first server is unavailable.
- AC5  — per-endpoint health probes: HTTP health check + GPU OOM error-pattern
         detection + slot capacity awareness.

All tests are hermetic (fake servers, mocked HTTP, no live llama-server).
"""

import asyncio
from types import SimpleNamespace

import pytest
from proxy.provider import (
    _check_local_backend_gpu_oom,
    _get_local_provider_endpoint,
)

from proxy import provider, router_helpers


def _clear_provider_state():
    """Reset module-global provider cooldown state between tests."""
    provider._provider_unavailable_until.clear()
    provider._attempted_providers.clear() if hasattr(provider, "_attempted_providers") else None


@pytest.fixture(autouse=True)
def _reset_provider_state():
    _clear_provider_state()
    yield
    _clear_provider_state()


# ---------------------------------------------------------------------------
# AC1: endpoint resolution for type: local providers
# ---------------------------------------------------------------------------

def test_local_provider_endpoint_explicit():
    """AC1: an explicit ``endpoint`` field is used verbatim (trailing slash stripped)."""
    cfg = {
        "name": "local-qwen3-server-1",
        "type": "local",
        "llama_model": "Qwen3",
        "endpoint": "http://192.168.0.199:8080/",
    }
    assert _get_local_provider_endpoint(cfg, {}) == "http://192.168.0.199:8080"


def test_local_provider_endpoint_defaults_to_localhost_port():
    """AC1 (backward compat): omitting ``endpoint`` uses localhost:{llama_server_port}."""
    cfg = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
    assert _get_local_provider_endpoint(cfg, {}) == "http://localhost:8080"
    config = {"server": {"llama_server_port": 9090}}
    assert _get_local_provider_endpoint(cfg, config) == "http://localhost:9090"


def test_local_provider_endpoint_empty_string_falls_back():
    """An empty ``endpoint`` field is treated as unset."""
    cfg = {"name": "local-qwen3", "type": "local", "endpoint": ""}
    assert _get_local_provider_endpoint(cfg, {}) == "http://localhost:8080"


# ---------------------------------------------------------------------------
# AC2: per-endpoint dispatch leases
# ---------------------------------------------------------------------------

def _make_dispatch_server():
    """Fresh fake server with dispatch-lease state (legacy flat dict)."""
    return SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 10}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )


@pytest.mark.asyncio
async def test_dispatch_lease_keyed_per_endpoint():
    """AC2: same session can hold an independent lease on two endpoints."""
    from proxy.router_helpers import (
        _dispatch_lease_key,
        _try_acquire_local_dispatch,
    )

    assert _dispatch_lease_key("http://a:8080", "sess-1") == ("http://a:8080", "sess-1")
    assert _dispatch_lease_key("http://b:8080", "sess-1") == ("http://b:8080", "sess-1")
    # Legacy single-server label keeps a plain session-id key.
    assert _dispatch_lease_key("local", "sess-1") == "sess-1"
    assert _dispatch_lease_key(None, "sess-1") == "sess-1"

    srv = _make_dispatch_server()
    # Acquire on server-a.
    acquired, owner, active, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-1", backend="http://a:8080"
    )
    assert acquired is True
    assert ("http://a:8080", "sess-1") in srv.local_dispatch_records

    # Same session on server-b is permitted (different endpoint, independent pool).
    acquired2, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-1", backend="http://b:8080"
    )
    assert acquired2 is True
    assert ("http://b:8080", "sess-1") in srv.local_dispatch_records


@pytest.mark.asyncio
async def test_dispatch_exhaustion_is_per_endpoint():
    """AC2: slot exhaustion on server-a does not block server-b."""
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = _make_dispatch_server()
    # Occupied record on server-a with a different session.
    srv.local_dispatch_records[("http://a:8080", "sess-owner")] = {
        "backend": "http://a:8080",
        "started_at": 1.0,
        "active": True,
        "expires_at": 10**12,
    }
    srv.local_active_queries = 1

    # Server-a is full (max_local=1, occupied by sess-owner).
    acquired_a, owner_a, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-new", backend="http://a:8080"
    )
    assert acquired_a is False
    assert owner_a == "sess-owner"

    # Server-b still has capacity — acquisition succeeds.
    acquired_b, _, _, _ = await _try_acquire_local_dispatch(
        srv, max_local=1, session_key="sess-new", backend="http://b:8080"
    )
    assert acquired_b is True


@pytest.mark.asyncio
async def test_release_and_cleanup_respect_endpoint_keys():
    """AC2: release/cleanup remove only the targeted endpoint's record."""
    from proxy.router_helpers import _cleanup_stale_local_dispatch, _release_local_dispatch

    srv = _make_dispatch_server()
    srv.local_dispatch_records[("http://a:8080", "sess-1")] = {
        "backend": "http://a:8080",
        "started_at": 1.0,
        "active": False,
        "expires_at": 1.0,  # expired inactive record
    }
    srv.local_dispatch_records[("http://b:8080", "sess-1")] = {
        "backend": "http://b:8080",
        "started_at": 1.0,
        "active": True,
        "expires_at": 10**12,
    }

    removed = await _cleanup_stale_local_dispatch(srv)
    assert removed == 1
    assert ("http://a:8080", "sess-1") not in srv.local_dispatch_records
    # The live record on server-b survives.
    assert ("http://b:8080", "sess-1") in srv.local_dispatch_records

    # Explicit release of the b record only.
    released = await _release_local_dispatch(srv, "sess-1", endpoint="http://b:8080")
    assert released is True
    assert ("http://b:8080", "sess-1") not in srv.local_dispatch_records


# ---------------------------------------------------------------------------
# AC3: per-endpoint slot save/restore paths
# ---------------------------------------------------------------------------

def test_slot_context_derives_per_endpoint_subdirectory(tmp_path):
    """AC3: slot files live under {session_slot_save_path}/{host}-{port}/."""
    from proxy.session import _build_slot_context, _slot_owners

    _slot_owners.clear()
    config = {
        "server": {
            "session_slot_save_path": str(tmp_path),
            "session_slot_pool_size": 2,
            "session_slot_timeout_seconds": 3.0,
        }
    }
    server_config = config["server"]

    slot_id, filename, _ = _build_slot_context(
        server_config, "sess-1", endpoint="http://192.168.0.199:8080"
    )
    assert slot_id is not None
    path = __import__("pathlib").Path(filename)
    assert path.parent.name == "192.168.0.199-8080"
    assert path.name == "slot_sess-1.bin"
    assert path.parent.exists()


def test_slot_context_backward_compat_no_endpoint(tmp_path):
    """AC3 (backward compat): no endpoint keeps the legacy root directory."""
    from proxy.session import _build_slot_context, _slot_owners

    _slot_owners.clear()
    config = {
        "server": {
            "session_slot_save_path": str(tmp_path),
            "session_slot_pool_size": 2,
        }
    }
    slot_id, filename, _ = _build_slot_context(config["server"], "sess-1")
    assert slot_id is not None
    path = __import__("pathlib").Path(filename)
    assert path.parent == tmp_path


def test_slot_registry_scoped_per_endpoint():
    """AC3: the same slot id can be assigned to different sessions on different endpoints."""
    from proxy.session import _assigned_slot_for_session, _slot_id_for_session, _slot_owners

    _slot_owners.clear()
    s1 = _slot_id_for_session("owner-a", 2, endpoint="http://a:8080")
    s2 = _slot_id_for_session("owner-b", 2, endpoint="http://b:8080")
    assert s1 == 0
    assert s2 == 0  # both servers assign their own slot 0
    assert _assigned_slot_for_session("owner-a", endpoint="http://a:8080") == 0
    assert _assigned_slot_for_session("owner-b", endpoint="http://b:8080") == 0
    assert _assigned_slot_for_session("owner-a", endpoint="http://b:8080") is None


# ---------------------------------------------------------------------------
# AC4: fallback chain across multiple local providers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def __getattr__(self, item):  # pragmatic: keep attribute access safe
        return None


@pytest.mark.asyncio
async def test_local_fallback_chain_tries_next_server(monkeypatch):
    """AC4: when server-1 is unavailable, the chain dispatches to server-2."""
    calls = []

    async def fake_proxy_to_local(req, path, endpoint=None):
        calls.append(endpoint)
        # Server-1 is unavailable (503), server-2 works.
        if endpoint and "199" in endpoint:
            return _FakeResponse(status_code=503, content=b"backend unavailable")
        resp = _FakeResponse(status_code=200, content=b"ok from server-2")
        return resp

    # Wire the fallback cycle's pointer to our fake.
    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_proxy_to_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote",
                        lambda: lambda req, path, cfg: _FakeResponse(status_code=503))

    model_config = {
        "providers": [
            {"name": "srv-a", "type": "local", "endpoint": "http://192.168.0.199:8080"},
            {"name": "srv-b", "type": "local", "endpoint": "http://192.168.0.200:8080"},
        ]
    }
    config = {"server": {"llama_server_port": 8080}}

    class _Req:
        headers = {}
        body = b"{'messages': [{'role': 'user', 'content': 'hi'}]}"

        async def body(self):
            return self.body

    result = await provider._proxy_with_fallback_cycle(
        _Req(), "v1/chat/completions", model_config, config
    )
    assert result.status_code == 200
    assert calls == ["http://192.168.0.199:8080", "http://192.168.0.200:8080"]


@pytest.mark.asyncio
async def test_local_provider_endpoint_passed_to_proxy_to_local(monkeypatch):
    """AC4: the resolved endpoint URL is passed to proxy_to_local."""
    seen = {}

    async def fake_proxy_to_local(req, path, endpoint=None):
        seen["endpoint"] = endpoint
        return _FakeResponse(status_code=200, content=b"ok")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_proxy_to_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote",
                        lambda: lambda req, path, cfg: _FakeResponse(status_code=503))

    model_config = {
        "providers": [
            {"name": "srv-a", "type": "local", "endpoint": "http://192.168.0.199:8080"},
        ]
    }
    config = {"server": {"llama_server_port": 8080}}

    class _Req:
        headers = {}
        body = b'{"messages": [{"role": "user", "content": "hi"}]}'

        async def body(self):
            return self.body

    result = await provider._proxy_with_fallback_cycle(
        _Req(), "v1/chat/completions", model_config, config
    )
    assert result.status_code == 200
    assert seen["endpoint"] == "http://192.168.0.199:8080"


# ---------------------------------------------------------------------------
# AC5: health probes — HTTP + GPU OOM + slot capacity
# ---------------------------------------------------------------------------

def test_gpu_oom_pattern_detection():
    """AC5: OOM error patterns are detected from response text."""
    from proxy.backend_health import _detect_gpu_oom_in_text

    assert _detect_gpu_oom_in_text("CUDA out of memory. Tried to allocate 1.2 GiB") is True
    assert _detect_gpu_oom_in_text("failed to allocate 2048 MB") is True
    assert _detect_gpu_oom_in_text("kv cache allocation failed") is True
    assert _detect_gpu_oom_in_text("everything is fine") is False
    assert _detect_gpu_oom_in_text("") is False


def test_check_local_backend_gpu_oom_helper():
    """AC5: provider helper wraps the OOM detector."""
    assert _check_local_backend_gpu_oom(
        "http://x:8080", '{"error": "CUDA out of memory"}'
    ) is True
    assert _check_local_backend_gpu_oom("http://x:8080", "ok") is False


@pytest.mark.asyncio
async def test_probe_local_backend_http_and_capacity(monkeypatch):
    """AC5: probe combines HTTP health + slot capacity awareness."""
    from proxy import backend_health

    async def fake_http(endpoint, timeout=5.0):
        return True

    async def fake_slots(endpoint, model_name=None, timeout=5.0):
        return 2, 4  # 2 available of 4 total

    monkeypatch.setattr(backend_health, "_probe_local_endpoint_http", fake_http)
    monkeypatch.setattr(backend_health, "_probe_local_slot_capacity", fake_slots)

    result = await backend_health.probe_local_backend(
        "http://192.168.0.199:8080", model_name="Qwen3"
    )
    assert result["http_ok"] is True
    assert result["available_slots"] == 2
    assert result["total_slots"] == 4
    assert result["capacity_ok"] is True
    assert result["gpu_oom"] is False


@pytest.mark.asyncio
async def test_probe_local_backend_unreachable(monkeypatch):
    """AC5: an unreachable endpoint fails open with unknown capacity."""
    from proxy import backend_health

    async def fake_http(endpoint, timeout=5.0):
        return False

    async def fake_slots(endpoint, model_name=None, timeout=5.0):
        return 0, 0

    monkeypatch.setattr(backend_health, "_probe_local_endpoint_http", fake_http)
    monkeypatch.setattr(backend_health, "_probe_local_slot_capacity", fake_slots)

    result = await backend_health.probe_local_backend("http://192.168.0.199:8080")
    assert result["http_ok"] is False
    assert result["available_slots"] == 0
    assert result["total_slots"] == 0
    assert result["capacity_ok"] is False

# ---------------------------------------------------------------------------
# Extra regression coverage for current-dev semantics (multi-backend port)
# ---------------------------------------------------------------------------

def test_endpoint_host_port_parsing():
    """URL → (host, port) derivation used for slot-path subdirectories."""
    from proxy.session import _endpoint_host_port

    assert _endpoint_host_port("http://192.168.0.199:8080") == ("192.168.0.199", 8080)
    assert _endpoint_host_port("http://localhost:8080") == ("localhost", 8080)
    assert _endpoint_host_port("http://my-host.org") == ("my-host.org", 8080)
    assert _endpoint_host_port("https://10.0.0.5:9090/") == ("10.0.0.5", 9090)
    # Unparseable → legacy default pair.
    assert _endpoint_host_port("not-a-url") == ("localhost", 8080)
    assert _endpoint_host_port("") == ("localhost", 8080)


def test_parse_endpoint_url():
    """router_helpers URL parser for per-endpoint dispatch records."""
    from proxy.router_helpers import _parse_endpoint_url

    assert _parse_endpoint_url("http://192.168.0.199:8080") == ("192.168.0.199", 8080)
    assert _parse_endpoint_url("http://localhost:9090/") == ("localhost", 9090)


def test_dispatch_key_session_id_roundtrip():
    """Session ids are recoverable from tuple and plain keys."""
    from proxy.router_helpers import _dispatch_key_session_id

    assert _dispatch_key_session_id(("http://a:8080", "sess-1")) == "sess-1"
    assert _dispatch_key_session_id("sess-1") == "sess-1"


@pytest.mark.asyncio
async def test_increment_decrement_respect_endpoint_records():
    """AC2: increment/decrement create and deactivate only the endpoint's record."""
    from proxy.router_helpers import (
        _decrement_local_active_queries,
        _increment_local_active_queries,
    )

    srv = _make_dispatch_server()
    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="http://a:8080", model_name="Qwen3"
    )
    assert ("http://a:8080", "sess-1") in srv.local_dispatch_records
    assert srv.local_dispatch_records[("http://a:8080", "sess-1")]["active"] is True
    assert srv.local_active_queries == 1

    # Decrement marks the endpoint record inactive (lease cooldown kept).
    await _decrement_local_active_queries(
        srv, session_key="sess-1", backend="http://a:8080"
    )
    import time as _t
    rec = srv.local_dispatch_records[("http://a:8080", "sess-1")]
    assert rec["active"] is False
    assert rec["expires_at"] > _t.monotonic()  # kept for lease timeout
    assert srv.local_active_queries == 0

    # A decrement for a *different* endpoint is a no-op on the a record.
    await _increment_local_active_queries(
        srv, session_key="sess-1", backend="http://b:8080", model_name="Qwen3"
    )
    await _decrement_local_active_queries(
        srv, session_key="sess-1", backend="http://a:8080"
    )
    assert srv.local_dispatch_records[("http://b:8080", "sess-1")]["active"] is True


def test_local_concurrency_info_pure_legacy_default_endpoint(monkeypatch):
    """Provider gate: legacy plain-key state uses the global generating counter.

    The fallback cycle always resolves a non-empty default endpoint for local
    providers without an explicit ``endpoint``. When the dispatch state is
    purely legacy (plain session-id keys), the per-endpoint count must not
    discard those records: the global generating counter gates exactly as it
    did before multi-backend support (LP-0MRPILSMW004T4H8).
    """
    import time as _time

    from proxy import provider

    class _Srv:
        local_dispatch_records = {
            "owner-1": {
                "backend": "local", "started_at": _time.monotonic(),
                "active": True, "expires_at": _time.monotonic() + 300,
            }
        }
        local_generating_queries = 1
        local_active_queries = 1

    # The helper reads state via ``import proxy.server``; patch its record
    # state directly.
    import proxy.server as srv_module
    monkeypatch.setattr(srv_module, "local_dispatch_records", _Srv.local_dispatch_records)
    monkeypatch.setattr(srv_module, "local_generating_queries", 1)
    monkeypatch.setattr(srv_module, "local_active_queries", 1)

    config = {"server": {"llama_server_port": 8080, "session_slot_pool_size": 1}}

    # Pure-legacy default endpoint → global generating counter (1 >= 1).
    cur, mx = provider._get_local_concurrency_info(config, endpoint="http://localhost:8080")
    assert (cur, mx) == (1, 1)

    # Explicitly endpoint-scoped query on a different server → 0 occupancy.
    cur2, _ = provider._get_local_concurrency_info(config, endpoint="http://10.0.0.9:8080")
    assert cur2 == 0

    # A non-default endpoint query is unaffected by plain-key legacy records.
    cur3, _ = provider._get_local_concurrency_info(
        config, endpoint="http://192.168.0.199:8080"
    )
    assert cur3 == 0


@pytest.mark.asyncio
async def test_local_concurrency_info_endpoint_records(monkeypatch):
    """Provider gate: per-endpoint counting when tuple-keyed records exist."""
    import time as _time

    import proxy.server as srv_module

    from proxy import provider

    srv_module.local_dispatch_records = {
        ("http://a:8080", "sess-x"): {
            "backend": "http://a:8080", "started_at": _time.monotonic(),
            "active": True, "expires_at": _time.monotonic() + 300,
        },
        ("http://b:8080", "sess-y"): {
            "backend": "http://b:8080", "started_at": _time.monotonic(),
            "active": True, "expires_at": _time.monotonic() + 300,
        },
    }
    monkeypatch.setattr(srv_module, "local_generating_queries", 2)
    monkeypatch.setattr(srv_module, "local_active_queries", 2)

    config = {"server": {"llama_server_port": 8080, "session_slot_pool_size": 1}}

    cur_a, mx = provider._get_local_concurrency_info(config, endpoint="http://a:8080")
    assert cur_a == 1  # only server-a's record counts
    cur_b, _ = provider._get_local_concurrency_info(config, endpoint="http://b:8080")
    assert cur_b == 1
