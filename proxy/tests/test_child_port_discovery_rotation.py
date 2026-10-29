"""Rotation-proof local child-port discovery (LP-0MVP7525P008RXMF AC1/AC5).

Context (F3 triage LP-0MTIHZ8M5005ZAU8, LP-0MTCMEV1G0022A35): the router
llama-server answers ``GET /slots`` with HTTP 500 whenever the poll lands while
the Qwen3 child is mid-giant-prefill, so every proxy-initiated ``/slots`` query
must target the child port directly.

The rank-1 fix added ``_discover_local_child_port()`` but it only understood the
startup spawn line (``spawning server instance with name=Qwen3 on port 58113``).
On 2026-10-26 the live ``llama-server.log`` was 9.6 MB with **zero** spawn lines
(they had been rotated away) and 38,761 live ``proxying request to model Qwen3 on
port 41395`` lines, so ``_is_sparse_log()`` said "real content", discovery
returned ``None``, and every caller silently fell back to the router — 1,754
router 500s in that window.

These tests pin the rotation-proof behaviour:

* the live ``proxying request ... on port`` signal is a discovery source;
* a miss is never pinned to ``None`` for the life of the pid (TTL-bounded);
* a positive result survives a later log mutation but is re-verified after its
  TTL so a child restart with the same parent pid cannot pin a stale port;
* model filtering (embed vs Qwen3) still applies to both sources.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from proxy import router_helpers as rh

PRODUCTION_SHAPE = (
    # No spawn line anywhere in this file (rotated away) — exactly the shape
    # observed in /var/log/llama-proxy/llama-server.log on 2026-10-26.
    "[59947] main: model loaded\n"
    "srv  proxy_reques: proxying request to model mxbai-embed on port 59947\n"
    "srv          load: prompt processing progress, n_tokens = 49152, progress = 0.406\n"
    "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
    "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
    "[41395] slot update_slots: id 0 | task 4 | n_past = 49152\n"
)


@pytest.fixture(autouse=True)
def _clear_child_port_caches():
    """Reset the module-level discovery caches between tests."""
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    yield
    rh._child_port_cache.clear()
    rh._child_port_cache_at.clear()
    rh._child_port_miss_cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0


def _srv(log_dir, pid=41395, model="Qwen3"):
    return SimpleNamespace(
        log_dir=Path(log_dir),
        llama_process=SimpleNamespace(pid=pid),
        current_model=model,
        config={"server": {"llama_server_port": 8080, "llama_router_mode": True}},
        logger=SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )


class TestRotationProofDiscovery:
    """AC1/AC5 — the live ``proxying request`` line is a discovery source."""

    def test_production_shape_rotated_log_yields_child_port(self, tmp_path):
        """Regression (AC5): rotated log, no spawn line, live proxy line.

        Fails against the pre-fix helper, which only scanned the spawn line and
        therefore returned None here (falling back to the router port 8080).
        """
        (tmp_path / "llama-server.log").write_text(PRODUCTION_SHAPE)
        srv = _srv(tmp_path)

        # The old (spawn-only) source genuinely has nothing to find here —
        # this is the exact reason discovery was inert in production.
        assert rh._scan_spawn_ports(tmp_path / "llama-server.log", "Qwen3") is None
        # ... and the fixed helper still resolves the live child port.
        assert rh._discover_local_child_port(srv) == 41395

    def test_live_source_filters_by_target_model(self, tmp_path):
        """The embed child's proxy line never wins for a Qwen3 target."""
        (tmp_path / "llama-server.log").write_text(PRODUCTION_SHAPE)
        srv = _srv(tmp_path)

        assert rh._discover_local_child_port(srv, model="Qwen3") == 41395
        rh._child_port_cache.clear()
        rh._child_port_cache_at.clear()
        assert rh._discover_local_child_port(srv, model="mxbai-embed") == 59947

    def test_newest_proxy_line_wins(self, tmp_path):
        """A model reload that moves the child port is picked up (last line)."""
        (tmp_path / "llama-server.log").write_text(
            "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
            "srv  proxy_reques: proxying request to model Qwen3 on port 46085\n"
        )
        assert rh._discover_local_child_port(_srv(tmp_path)) == 46085

    def test_fresh_spawn_line_still_works(self, tmp_path):
        """A fresh log with the startup spawn line (first poll after restart)."""
        (tmp_path / "llama-server.log").write_text(
            "srv          load: spawning server instance with name=Qwen3 on port 58113\n"
        )
        assert rh._discover_local_child_port(_srv(tmp_path)) == 58113

    def test_sparse_primary_falls_back_to_rotated_spawn_line(self, tmp_path):
        """NUL-padded current log + rotated log holding the spawn line."""
        (tmp_path / "llama-server.log").write_bytes(b"\x00" * 4096)
        (tmp_path / "llama-server.1.log").write_text(
            "srv          load: spawning server instance with name=Qwen3 on port 47779\n"
        )
        assert rh._discover_local_child_port(_srv(tmp_path)) == 47779

    def test_sparse_primary_falls_back_to_rotated_proxy_line(self, tmp_path):
        """Rotated log holding only proxy lines is still a usable source."""
        (tmp_path / "llama-server.log").write_bytes(b"\x00" * 4096)
        (tmp_path / "llama-server.1.log").write_text(
            "srv  proxy_reques: proxying request to model Qwen3 on port 52222\n"
        )
        assert rh._discover_local_child_port(_srv(tmp_path)) == 52222

    def test_missing_log_returns_none(self, tmp_path):
        """No log at all -> None (caller falls back to the router)."""
        assert rh._discover_local_child_port(_srv(tmp_path)) is None

    def test_unrelated_content_returns_none(self, tmp_path):
        """Real content without a child-port signal -> None, no global fallback."""
        (tmp_path / "llama-server.log").write_text("[58113] main: model loaded\n")
        srv = _srv(tmp_path, pid=9023)
        assert rh._discover_local_child_port(srv) is None


class TestMissIsNotPinnedForThePidLifetime:
    """The 2026-10-26 regression: a ``None`` cached for the whole pid life."""

    def test_signal_appearing_after_a_miss_is_picked_up(self, tmp_path, monkeypatch):
        """A miss is TTL-bounded, not permanent — a later signal resolves.

        Cold start: neither source exists yet. The very first proxied request
        writes the live line, and the next poll (after the miss TTL) must find
        it instead of serving ``None`` for the rest of the process lifetime.
        """
        log = tmp_path / "llama-server.log"
        log.write_text("[41395] main: model loaded\n")
        srv = _srv(tmp_path, pid=41395)
        assert rh._discover_local_child_port(srv) is None  # cached miss

        log.write_text(
            "[41395] main: model loaded\n"
            "srv  proxy_reques: proxying request to model Qwen3 on port 41395\n"
        )
        # Still cached as a miss (within the TTL)...
        monkeypatch.setenv("CHILD_PORT_MISS_TTL", "30")
        assert rh._discover_local_child_port(srv) is None
        # ...but after the TTL it re-scans and resolves the child port.
        monkeypatch.setenv("CHILD_PORT_MISS_TTL", "0")
        assert rh._discover_local_child_port(srv) == 41395

    def test_miss_ttl_avoids_rescanning_while_no_signal(self, tmp_path):
        """A cached miss does not re-read the log on every poll (cheapness)."""
        log = tmp_path / "llama-server.log"
        log.write_text("[41395] main: model loaded\n")
        srv = _srv(tmp_path, pid=41395)
        assert rh._discover_local_child_port(srv) is None

        calls = {"n": 0}
        real = rh._scan_child_port_signals

        def _counting(path, target):
            calls["n"] += 1
            return real(path, target)

        rh._scan_child_port_signals = _counting
        try:
            for _ in range(5):
                assert rh._discover_local_child_port(srv) is None
        finally:
            rh._scan_child_port_signals = real
        assert calls["n"] == 0, "a cached miss must not re-scan the log"


class TestPositiveCacheAndReverification:
    """Positives are cached, re-verified on TTL, and never regress."""

    def test_positive_is_cached(self, tmp_path):
        """Repeated calls do not re-read the log."""
        log = tmp_path / "llama-server.log"
        log.write_text(PRODUCTION_SHAPE)
        srv = _srv(tmp_path)
        assert rh._discover_local_child_port(srv) == 41395

        log.write_text("garbage\n")  # mutated after the positive was cached
        assert rh._discover_local_child_port(srv) == 41395

    def test_new_pid_reparses(self, tmp_path):
        """A llama-server restart (new pid) re-discovers the child port."""
        log = tmp_path / "llama-server.log"
        log.write_text("srv  proxy_reques: proxying request to model Qwen3 on port 41395\n")
        srv = _srv(tmp_path, pid=111)
        assert rh._discover_local_child_port(srv) == 41395

        log.write_text("srv  proxy_reques: proxying request to model Qwen3 on port 46085\n")
        srv.llama_process = SimpleNamespace(pid=222)
        assert rh._discover_local_child_port(srv) == 46085

    def test_model_change_reparses(self, tmp_path):
        """The cache is keyed by model, so a model switch re-discovers."""
        (tmp_path / "llama-server.log").write_text(PRODUCTION_SHAPE)
        srv = _srv(tmp_path)
        assert rh._discover_local_child_port(srv) == 41395
        assert rh._discover_local_child_port(srv, model="mxbai-embed") == 59947

    def test_ttl_expiry_reverifies_and_updates(self, tmp_path, monkeypatch):
        """After the TTL a moved child port is re-discovered (restart-proof)."""
        log = tmp_path / "llama-server.log"
        log.write_text("srv  proxy_reques: proxying request to model Qwen3 on port 41395\n")
        srv = _srv(tmp_path, pid=41395)
        assert rh._discover_local_child_port(srv) == 41395

        log.write_text("srv  proxy_reques: proxying request to model Qwen3 on port 46085\n")
        monkeypatch.setenv("CHILD_PORT_CACHE_TTL", "0")
        assert rh._discover_local_child_port(srv) == 46085

    def test_ttl_expiry_keeps_last_known_good_port(self, tmp_path, monkeypatch):
        """Re-verification that finds nothing keeps the cached port, never None."""
        log = tmp_path / "llama-server.log"
        log.write_text("srv  proxy_reques: proxying request to model Qwen3 on port 41395\n")
        srv = _srv(tmp_path, pid=41395)
        assert rh._discover_local_child_port(srv) == 41395

        log.write_text("[41395] main: model loaded\n")  # signal rotated away again
        monkeypatch.setenv("CHILD_PORT_CACHE_TTL", "0")
        assert rh._discover_local_child_port(srv) == 41395

    def test_positive_cache_timestamp_advances_on_reverify(self, tmp_path, monkeypatch):
        """A no-change re-verification refreshes the timestamp (no re-scan storm)."""
        (tmp_path / "llama-server.log").write_text(PRODUCTION_SHAPE)
        srv = _srv(tmp_path, pid=41395)
        assert rh._discover_local_child_port(srv) == 41395
        key = (41395, "qwen3")
        first = rh._child_port_cache_at[key]

        time.sleep(0.01)
        monkeypatch.setenv("CHILD_PORT_CACHE_TTL", "0")
        assert rh._discover_local_child_port(srv) == 41395
        assert rh._child_port_cache_at[key] > first
