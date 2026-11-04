#!/usr/bin/env python3
"""Eviction saves must use the adaptive timeout, and failures must say which
path issued them (LP-0MVVX6RM3001P4EY).

Measured evidence (39.17 h window 2026-10-29T12:58:32 → 2026-10-31T04:09:00,
``scripts/attribute_slot_save_failures.py``):

- router path: 46 ok / 1 failed (98 %);
- eviction path: 60 ok / **93 failed**, and **93/93** failures used
  ``timeout=3.0s`` — the static base — even though ``proxy/config.yaml``
  deploys ``session_slot_timeout_per_token_seconds: 0.0015`` and
  ``session_slot_max_timeout_seconds: 60``;
- 11 of the 94 failures were logged with zero active queries, so a 3 s bound
  alone cannot cover the KV-cache serialisation.

Root cause: ``plan_eviction_snapshot`` calls ``_build_slot_context(..., None,
...)`` — no request body — so the token estimate is 0 and the per-token
scaling contributes nothing to the timeout. The session's recorded prompt
prefix (the same trustworthy count the sidecar writer uses) is the estimate
that path should feed in.

The second gap is attribution: ``slot_save failed`` lines carried no
``source=``, so the earlier triage counted 94 eviction failures as
"router-path" failures. These tests pin both halves of the fix.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from proxy.provider import _last_prompt_prefix, record_prompt_prefix
from proxy.session import (
    _build_slot_context,
    _call_slot_endpoint,
    _save_slot_snapshot,
    _slot_owners,
    plan_eviction_snapshot,
    save_evicted_session_snapshot,
)

MODEL = "Qwen3"
SESSION = "sess-evict-timeout"
ENDPOINT = "http://localhost:8080"

# The deployed values (proxy/config.yaml) — the fix must reuse this policy,
# not invent a new tuning value.
BASE_TIMEOUT = 3.0
PER_TOKEN = 0.0015
MAX_TIMEOUT = 60.0


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the process-global slot registry / recorded prefix between tests."""
    _slot_owners.clear()
    _last_prompt_prefix.clear()
    yield
    _slot_owners.clear()
    _last_prompt_prefix.clear()


def _config(tmp_path) -> dict:
    slot_root = tmp_path / "slot-cache"
    slot_root.mkdir(parents=True, exist_ok=True)
    return {
        "session_slot_save_path": str(slot_root),
        "session_slot_pool_size": 2,
        "session_slot_timeout_seconds": BASE_TIMEOUT,
        "session_slot_timeout_per_token_seconds": PER_TOKEN,
        "session_slot_max_timeout_seconds": MAX_TIMEOUT,
    }


def _messages(count: int = 3) -> list[dict]:
    return [{"role": "user", "content": f"turn {i}"} for i in range(count)]


class _Srv:
    """Minimal stand-in for the global proxy server."""

    def __init__(self, logger=None):
        self.current_model = MODEL
        self.logger = logger or logging.getLogger("test-slot-source")
        self._http_client = None


class _TimeoutClient:
    async def post(self, *_args, **_kwargs):
        raise httpx.ReadTimeout("read timeout")

    async def aclose(self):
        pass


class _OkClient:
    def __init__(self):
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return httpx.Response(200, request=httpx.Request("POST", "http://x"))


# ---------------------------------------------------------------------------
# AC: the eviction path gets the adaptive timeout from the recorded prefix
# ---------------------------------------------------------------------------


def test_build_slot_context_scales_with_an_explicit_estimate(tmp_path):
    """A body-less caller can still supply the token estimate (the fix)."""
    _slot, _filename, timeout = _build_slot_context(
        _config(tmp_path), SESSION, None, estimated_tokens=20_000
    )

    assert timeout == pytest.approx(BASE_TIMEOUT + PER_TOKEN * 20_000)


def test_build_slot_context_caps_the_explicit_estimate(tmp_path):
    _slot, _filename, timeout = _build_slot_context(
        _config(tmp_path), SESSION, None, estimated_tokens=200_000
    )

    assert timeout == pytest.approx(MAX_TIMEOUT)


def test_build_slot_context_keeps_the_base_timeout_without_an_estimate(tmp_path):
    """Pins the pre-fix behaviour for a genuinely unknown context size."""
    _slot, _filename, timeout = _build_slot_context(_config(tmp_path), SESSION, None)

    assert timeout == pytest.approx(BASE_TIMEOUT)


def test_eviction_plan_uses_the_recorded_prompt_prefix(tmp_path, monkeypatch):
    """The eviction plan must not fall back to the 3 s base for a known prefix."""
    from proxy import session as session_mod

    srv = _Srv()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    # The session must already own a slot (an eviction never allocates one).
    assert session_mod._slot_id_for_session(SESSION, 2, endpoint=ENDPOINT) is not None
    record_prompt_prefix(MODEL, SESSION, 20_000, _messages())
    config = _config(tmp_path)

    plan = plan_eviction_snapshot(SESSION, config, endpoint=ENDPOINT)

    assert plan is not None, "an eviction with a slot assignment must plan a save"
    _slot_id, _filename, timeout = plan
    assert timeout == pytest.approx(BASE_TIMEOUT + PER_TOKEN * 20_000)


def test_eviction_plan_without_a_recorded_prefix_keeps_the_base_timeout(
    tmp_path, monkeypatch
):
    from proxy import session as session_mod

    srv = _Srv()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    assert session_mod._slot_id_for_session(SESSION, 2, endpoint=ENDPOINT) is not None
    config = _config(tmp_path)

    plan = plan_eviction_snapshot(SESSION, config, endpoint=ENDPOINT)

    assert plan is not None
    _slot_id, _filename, timeout = plan
    assert timeout == pytest.approx(BASE_TIMEOUT)


def test_eviction_save_forwards_its_source(tmp_path, monkeypatch):
    """The eviction caller tags the save so failures are attributable."""
    from proxy import session as session_mod

    srv = _Srv()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    seen = {}

    async def _fake_save(_port, _slot, target, timeout, model=None, endpoint=None, source=None):
        seen.update(target=target, timeout=timeout, source=source)
        return False

    monkeypatch.setattr(session_mod, "_save_slot_snapshot", _fake_save)
    assert session_mod._slot_id_for_session(SESSION, 2, endpoint=ENDPOINT) is not None
    record_prompt_prefix(MODEL, SESSION, 20_000, _messages())
    config = _config(tmp_path)
    plan = plan_eviction_snapshot(SESSION, config, endpoint=ENDPOINT)

    asyncio.run(
        save_evicted_session_snapshot(SESSION, config, endpoint=ENDPOINT, plan=plan)
    )

    assert seen["source"] == "eviction"
    assert seen["timeout"] == pytest.approx(BASE_TIMEOUT + PER_TOKEN * 20_000)


# ---------------------------------------------------------------------------
# AC: failure logs name the save path (the attribution gap behind the
# "66 % router-path failures" mismeasurement)
# ---------------------------------------------------------------------------


def test_save_failure_log_names_the_source_path(tmp_path, monkeypatch, caplog):
    from proxy import session as session_mod

    srv = _Srv()
    srv._http_client = _TimeoutClient()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    caplog.set_level(logging.WARNING)

    ok = asyncio.run(
        _save_slot_snapshot(
            8080,
            0,
            str(tmp_path / "slot_x.bin"),
            BASE_TIMEOUT,
            source="eviction",
        )
    )

    assert ok is False
    assert "slot_save failed" in caplog.text
    assert "source=eviction" in caplog.text


def test_default_source_is_explicitly_unknown(tmp_path, monkeypatch, caplog):
    """A caller that forgets ``source=`` must not silently look like a router save."""
    from proxy import session as session_mod

    srv = _Srv()
    srv._http_client = _TimeoutClient()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    caplog.set_level(logging.WARNING)

    asyncio.run(
        _call_slot_endpoint(
            8080, 1, "save", str(tmp_path / "slot_y.bin"), BASE_TIMEOUT
        )
    )

    assert "source=unknown" in caplog.text


def test_save_success_debug_log_names_the_source(tmp_path, monkeypatch, caplog):
    from proxy import session as session_mod

    srv = _Srv()
    srv._http_client = _OkClient()
    monkeypatch.setattr(session_mod, "_srv", lambda: srv)
    caplog.set_level(logging.DEBUG)

    ok = asyncio.run(
        _save_slot_snapshot(
            8080,
            0,
            str(tmp_path / "slot_z.bin"),
            BASE_TIMEOUT,
            source="router",
        )
    )

    assert ok is True
    assert "slot_save ok" in caplog.text
    assert "source=router" in caplog.text


def test_source_is_forwarded_to_the_endpoint_call(monkeypatch):
    from proxy import session as session_mod

    seen = {}

    async def _fake_call(_port, _slot, action, filename, timeout, model=None, endpoint=None, source=None):
        seen["source"] = source
        return True

    monkeypatch.setattr(session_mod, "_call_slot_endpoint", _fake_call)

    ok = asyncio.run(
        _save_slot_snapshot(8080, 0, "/tmp/slot_q.bin", BASE_TIMEOUT, source="router")
    )

    assert ok is True
    assert seen["source"] == "router"
