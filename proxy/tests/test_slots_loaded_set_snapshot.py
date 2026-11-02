"""Synchronously readable loaded-set over the cached ``/models`` id snapshot.

Slice 2 of LP-0MVVCY3HQ0009F0K (chat-model child-port fallback for unloaded
local models), delivered by LP-0MVWBX2YT001OEYE. The consumer
(``_resolve_local_slots_target()``) is synchronous while
``srv.router_is_model_loaded()`` is async, so the guard must read a cache,
never await — and must not add a second ``/models`` cache (Q3 = B): the loaded
set reuses LP-0MVU7ZZIA0092AXW's cached id snapshot.

Pinned here:

* ``_loaded_models_snapshot()`` returns the cached loaded-id set with **zero**
  ``/models`` HTTP requests when the snapshot is warm (request-counting fake);
* it returns ``None`` (indeterminate) when the probe never succeeded, the
  last probe errored / returned a non-200, or the payload is unparseable —
  and never an empty set for an unknown state;
* the loaded set is served from the **same** snapshot dict as the id set
  (one ``/models`` request per TTL, not one per view);
* a llama-router restart (parent pid change) drops the snapshot, and a model
  switch is reflected within one TTL (``_child_port_cache_ttl()``);
* ``is_model_known_unloaded(name)`` matches case-insensitively and returns
  ``None`` for an unresolved (empty) name.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from proxy import router_helpers as rh

ROUTER_PORT = 8080

#: Production-shaped /models payload: Qwen3-32B and the embed child are
#: loaded, gpt120 is positively unloaded.
DEFAULT_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "Qwen3-32B", "status": {"value": "loaded"}},
        {"id": "mxbai-embed", "status": {"value": "loaded"}},
        {"id": "gpt120", "status": {"value": "unloaded"}},
    ],
}

_UNSET = object()
"""Sentinel so an explicit ``payload=None`` (the non-200 shape) is distinct
from "use the default payload"."""


def _srv(
    tmp_path,
    *,
    pid: int = 41395,
    current_model: str = "Qwen3",
    payload=_UNSET,
    raise_on_list: bool = False,
):
    """Minimal router-mode server handle with a stubbed, counting lister."""
    calls = {"models": 0}

    def _warning(message, *args, **kwargs):
        pass

    srv = SimpleNamespace(
        log_dir=Path(tmp_path),
        llama_process=SimpleNamespace(pid=pid),
        current_model=current_model,
        config={
            "server": {
                "llama_server_port": ROUTER_PORT,
                "llama_router_mode": True,
            }
        },
        logger=SimpleNamespace(warning=_warning, debug=lambda *a, **k: None),
        models_calls=calls,
    )
    srv.payload = DEFAULT_PAYLOAD if payload is _UNSET else payload
    srv.raise_on_list = raise_on_list

    async def _router_list_models():
        calls["models"] += 1
        if srv.raise_on_list:
            raise RuntimeError("probe failed")
        return srv.payload

    srv.router_list_models = _router_list_models
    return srv


@pytest.fixture(autouse=True)
def _clear_loaded_set_caches():
    """Reset the module-level /models snapshot caches between tests."""
    rh._router_model_ids_snapshot.clear()
    rh._child_model_signal_cache.clear()
    yield
    rh._router_model_ids_snapshot.clear()
    rh._child_model_signal_cache.clear()


def _age_snapshot(seconds: float) -> None:
    """Push the snapshot's fetch timestamp into the past (TTL-expiry sim)."""
    rh._router_model_ids_snapshot["at"] = time.monotonic() - seconds


# ---------------------------------------------------------------------------
# AC1: warm snapshot -> loaded set with zero /models HTTP requests
# ---------------------------------------------------------------------------


class TestWarmSnapshotZeroHttp:
    @pytest.mark.asyncio
    async def test_warm_loaded_snapshot_costs_zero_models_requests(
        self, tmp_path
    ):
        """AC1: the loaded set comes from the cache; no refetch while warm."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)
        assert srv.models_calls["models"] == 1

        for _ in range(25):
            assert rh._loaded_models_snapshot(srv) == {
                "Qwen3-32B",
                "mxbai-embed",
            }
            assert rh.is_model_known_unloaded(srv, "gpt120") is True

        assert srv.models_calls["models"] == 1, (
            "a warm loaded-set read must never trigger a /models request"
        )

    @pytest.mark.asyncio
    async def test_loaded_set_derives_from_the_id_snapshot(self, tmp_path):
        """AC3: both views share the one cached /models snapshot dict."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)

        assert rh._loaded_models_snapshot(srv) == set(
            rh._router_model_ids_snapshot["loaded"]
        )
        # One fetch feeds both the id resolution and the loaded set.
        assert rh._cached_router_model_ids(srv) == [
            "Qwen3-32B",
            "mxbai-embed",
            "gpt120",
        ]
        assert srv.models_calls["models"] == 1


# ---------------------------------------------------------------------------
# AC2: indeterminate (None) on failed / unparseable probes — never empty
# ---------------------------------------------------------------------------


class TestIndeterminateStates:
    def test_never_probed_returns_none(self, tmp_path):
        """No successful probe yet -> None, not an empty set."""
        srv = _srv(tmp_path)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "gpt120") is None

    @pytest.mark.asyncio
    async def test_probe_error_returns_none(self, tmp_path):
        """The lister raising leaves no trustworthy loaded set."""
        srv = _srv(tmp_path, raise_on_list=True)
        assert await rh.ensure_router_model_ids(srv) == []
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is None

    @pytest.mark.asyncio
    async def test_non_200_payload_returns_none(self, tmp_path):
        """A None payload (non-200 / swallowed error) -> None, never set()."""
        srv = _srv(tmp_path, payload=None)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "gpt120") is None

    @pytest.mark.asyncio
    async def test_unparseable_payload_returns_none(self, tmp_path):
        """A non-dict or non-list payload is unparseable -> None."""
        srv = _srv(tmp_path, payload=["not", "a", "dict"])
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) is None
        assert rh.is_model_known_unloaded(srv, "Qwen3-32B") is None

        srv2 = _srv(tmp_path, payload={"data": "nope"})
        await rh.ensure_router_model_ids(srv2)
        assert rh._loaded_models_snapshot(srv2) is None

    @pytest.mark.asyncio
    async def test_known_empty_payload_is_a_known_empty_set(self, tmp_path):
        """A 200 with an empty data list positively reports nothing loaded."""
        srv = _srv(tmp_path, payload={"object": "list", "data": []})
        loaded = await rh.ensure_loaded_models_snapshot(srv)
        assert loaded == set()
        assert rh.is_model_known_unloaded(srv, "gpt120") is True


# ---------------------------------------------------------------------------
# AC4: invalidation — restart (pid change), TTL expiry, model switch
# ---------------------------------------------------------------------------


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_pid_change_drops_the_snapshot_and_refreshes(self, tmp_path):
        """A llama-router restart (new parent pid) invalidates the loaded set."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) == {"Qwen3-32B", "mxbai-embed"}

        srv.llama_process = SimpleNamespace(pid=999)
        srv.payload = {
            "object": "list",
            "data": [{"id": "Qwen2.5-7B", "status": {"value": "loaded"}}],
        }
        assert rh._loaded_models_snapshot(srv) is None

        await rh.ensure_router_model_ids(srv)
        assert srv.models_calls["models"] == 2
        assert rh._loaded_models_snapshot(srv) == {"Qwen2.5-7B"}

    @pytest.mark.asyncio
    async def test_ttl_expiry_is_indeterminate_then_refreshes(self, tmp_path):
        """Stale data fails open; the async refresh repopulates the set."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) == {"Qwen3-32B", "mxbai-embed"}

        _age_snapshot(rh._child_port_cache_ttl() + 100)
        assert rh._loaded_models_snapshot(srv) is None, (
            "an expired snapshot must fail open, never a stale loaded set"
        )

        srv.payload = {
            "object": "list",
            "data": [{"id": "Qwen2.5-7B", "status": {"value": "loaded"}}],
        }
        loaded = await rh.ensure_loaded_models_snapshot(srv)
        assert loaded == {"Qwen2.5-7B"}
        assert srv.models_calls["models"] == 2

    @pytest.mark.asyncio
    async def test_ttl_consistent_with_child_port_cache_ttl(self, tmp_path, monkeypatch):
        """A longer child-port TTL keeps the loaded set readable longer."""
        monkeypatch.setenv("CHILD_PORT_CACHE_TTL", "1000")
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)

        _age_snapshot(400)  # beyond the default 300s, inside the 1000s TTL
        assert rh._loaded_models_snapshot(srv) == {"Qwen3-32B", "mxbai-embed"}

    @pytest.mark.asyncio
    async def test_model_switch_reflected_within_one_ttl(self, tmp_path):
        """A changed current_model / /models listing is visible after refresh."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)
        assert rh._loaded_models_snapshot(srv) == {"Qwen3-32B", "mxbai-embed"}

        srv.current_model = "Qwen2.5-7B"
        srv.payload = {
            "object": "list",
            "data": [{"id": "Qwen2.5-7B", "status": {"value": "loaded"}}],
        }
        assert rh._loaded_models_snapshot(srv) is None

        await rh.ensure_loaded_models_snapshot(srv)
        assert rh._loaded_models_snapshot(srv) == {"Qwen2.5-7B"}


# ---------------------------------------------------------------------------
# AC5: is_model_known_unloaded — case-insensitive, tri-state membership
# ---------------------------------------------------------------------------


class TestIsModelKnownUnloaded:
    @pytest.mark.asyncio
    async def test_case_insensitive_membership(self, tmp_path):
        """Loaded matching is case-insensitive; gpt120 is definitively not."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)

        for name in ("Qwen3-32B", "qwen3-32b", "QWEN3-32B"):
            assert rh.is_model_known_unloaded(srv, name) is False, name
        for name in ("gpt120", "GPT120"):
            assert rh.is_model_known_unloaded(srv, name) is True, name

    @pytest.mark.asyncio
    async def test_empty_name_is_indeterminate(self, tmp_path):
        """An unresolved (empty) name fails open -> None, never a skip."""
        srv = _srv(tmp_path)
        await rh.ensure_router_model_ids(srv)

        assert rh.is_model_known_unloaded(srv, "") is None
        assert rh.is_model_known_unloaded(srv, None) is None

    @pytest.mark.asyncio
    async def test_presence_without_status_counts_as_loaded(self, tmp_path):
        """Same rule as router_is_model_loaded: presence without status."""
        srv = _srv(
            tmp_path,
            payload={
                "object": "list",
                "data": [
                    {"id": "no-status-model"},
                    {"id": "ready-model", "status": {"value": "ready"}},
                    {"id": "running-model", "status": {"value": "running"}},
                    {"id": "active-model", "status": {"value": "active"}},
                    {"id": "loading-model", "status": {"value": "loading"}},
                    {"id": "error-model", "status": {"value": "error"}},
                ],
            },
        )
        await rh.ensure_router_model_ids(srv)

        assert rh._loaded_models_snapshot(srv) == {
            "no-status-model",
            "ready-model",
            "running-model",
            "active-model",
        }
        assert rh.is_model_known_unloaded(srv, "no-status-model") is False
        assert rh.is_model_known_unloaded(srv, "loading-model") is True
        assert rh.is_model_known_unloaded(srv, "error-model") is True


# ---------------------------------------------------------------------------
# The two "is it loaded" views cannot disagree (shared readiness rules)
# ---------------------------------------------------------------------------


#: Every readiness shape the router builds have been observed to emit.
READINESS_SHAPES = {
    "presence-no-status": {},
    "loaded": {"status": {"value": "loaded"}},
    "ready": {"status": {"value": "ready"}},
    "running": {"status": {"value": "running"}},
    "active": {"status": {"value": "active"}},
    "loading": {"status": {"value": "loading"}},
    "unloaded": {"status": {"value": "unloaded"}},
    "error": {"status": {"value": "error"}},
    "failed": {"status": {"value": "failed"}},
    "status-string-loaded": {"status": "Loaded"},
    "status-string-unloaded": {"status": "unloaded"},
    "unknown-status-value": {"status": {"value": "mystery"}},
    "unknown-status-shape": {"status": 7},
}


class TestViewsAgree:
    @pytest.mark.asyncio
    async def test_async_and_sync_views_agree_on_every_readiness_shape(
        self, tmp_path, monkeypatch
    ):
        """AC5: the shared readiness rules keep both views in lockstep.

        The async ``router_is_model_loaded()`` and the synchronous
        ``is_model_known_unloaded()`` must never disagree for the same
        ``/models`` payload — a disagreement would either skip a loaded model's
        poll or poll an unloaded one.
        """
        from proxy import lifecycle

        payload = {
            "object": "list",
            "data": [
                {"id": name, **shape} for name, shape in READINESS_SHAPES.items()
            ],
        }
        srv = _srv(tmp_path, payload=payload)
        await rh.ensure_router_model_ids(srv)

        async def _list_models():
            return payload

        monkeypatch.setattr(
            lifecycle,
            "_srv",
            lambda: SimpleNamespace(router_list_models=_list_models),
        )

        for name in READINESS_SHAPES:
            async_loaded = await lifecycle.router_is_model_loaded(name)
            known_unloaded = rh.is_model_known_unloaded(srv, name)
            assert known_unloaded is not None, name
            assert known_unloaded is (not async_loaded), (
                f"views disagree for {name}: async_loaded={async_loaded}, "
                f"known_unloaded={known_unloaded}"
            )
