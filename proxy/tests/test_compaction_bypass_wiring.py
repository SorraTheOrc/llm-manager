"""
Bypass-path compaction wiring tests (LP-0MU5ARWSP001BYYB child LP-0MU5ICX7H007QKVC).

Verifies that ``_proxy_with_fallback_cycle`` evaluates server-side compaction
in the local-bypass path and re-enables local dispatch when compaction brings
the session back under the thresholds.

ACs covered:
  AC1 — compaction evaluated before the bypass ``continue``.
  AC2 — successful compaction re-enables local on the same request.
  AC3 — noop/failure preserves the original bypass behaviour.
  AC4 — the outcome is observable in logs.
  AC5 — exactly one local dispatch (no double dispatch).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import proxy.provider as provider
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _Resp:
    def __init__(self, status_code: int = 200, content: bytes = b"ok"):
        self.status_code = status_code
        self.body = content
        self.headers = {}


def _messages(n: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


async def _remote_ok(req, path, cfg):
    return _Resp(200, json.dumps({"choices": [{"message": {"content": "remote"}}]}).encode())


async def _remote_fail(req, path, cfg):
    return _Resp(502, b"remote unavailable")


@pytest.fixture(autouse=True)
def reset_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    yield


@pytest.fixture(autouse=True)
def pin_fast_mode(monkeypatch):
    monkeypatch.setattr("proxy.mode.read_mode", lambda: "fast")


def _model_config(with_local: bool = True) -> dict:
    providers = []
    if with_local:
        providers.append({"name": "local-llama", "type": "local", "llama_model": "Qwen3"})
    providers.append(
        {"name": "remote-primary", "type": "remote", "endpoint": "https://api.example.com/v1"}
    )
    return {"providers": providers}


_CFG = {"server": {"llama_server_port": 8080, "local_model_ctx_size": 262144}}


# ---------------------------------------------------------------------------
# AC1 / AC4 — compaction evaluated on bypass, outcome observable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_evaluated_on_bypass(monkeypatch, caplog):
    """AC1/AC4: an oversized bypass evaluates compaction and logs the outcome."""
    calls = []

    monkeypatch.setattr(provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large"))

    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        calls.append({"messages": list(msgs)})
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": msgs, "reason": "below_trigger",
        }

    local_calls = []

    async def fake_local(req, path, endpoint=None):
        local_calls.append(True)
        return _Resp(200)

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_fail)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            with pytest.raises(provider.ChainExhaustedError):
                await provider._proxy_with_fallback_cycle(
                    _FakeRequest(json.dumps({"model": "test", "messages": _messages(40)}).encode()),
                    "v1/chat/completions", _model_config(), _CFG,
                )

    assert len(calls) == 1, "compaction must be evaluated on the bypass path"
    assert local_calls == [], "noop compaction must not dispatch local"
    assert any("compaction_bypass_eval" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# AC2 / AC5 — successful compaction re-enables local with compacted body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_compaction_reenables_local_with_compacted_body(monkeypatch, caplog):
    """AC2/AC5: a live compaction that brings the session under the threshold
    dispatches local exactly once with the compacted history."""
    compacted = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "<summary>...</summary>"},
        {"role": "assistant", "content": "recent"},
    ]

    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        return {
            "action": "compact", "applied": True, "dry_run": False,
            "messages": compacted,
            "estimated_before": 120000, "estimated_after": 2000,
            "reason": "trigger_exceeded",
            "summary_text": "...", "turns_summarized": 38, "recent_turns_kept": 2,
        }

    # First call (bypass decision) skips; the post-compaction re-check passes.
    skip_returns = [(True, "context_too_large"), (False, None)]

    def fake_skip(*a, **k):
        return skip_returns.pop(0) if skip_returns else (False, None)

    monkeypatch.setattr(provider, "_should_skip_local", fake_skip)

    local_bodies = []

    async def fake_local(req, path, endpoint=None):
        body = await req.body()
        local_bodies.append(json.loads(body))
        return _Resp(200, b'{"choices":[{"message":{"content":"ok"}}]}')

    remote_calls = []

    async def fake_remote(req, path, cfg):
        remote_calls.append(cfg["name"])
        return _Resp(200, b"remote")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            result = await provider._proxy_with_fallback_cycle(
                _FakeRequest(json.dumps({"model": "test", "messages": _messages(40)}).encode()),
                "v1/chat/completions", _model_config(), _CFG,
            )

    assert result.status_code == 200
    assert len(local_bodies) == 1, "exactly one local dispatch after compaction"
    assert local_bodies[0]["messages"] == compacted, "compacted history must be dispatched"
    assert remote_calls == [], "remote must not be used after a successful rescue"
    assert any("routing_compaction_local_redispatch" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# AC3 — noop / failure preserves the original bypass behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_still_oversized_falls_back_to_remote(monkeypatch):
    """AC3: when the compacted session is still over the threshold, the
    request continues to the remote provider (no local dispatch)."""
    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        return {
            "action": "compact", "applied": True, "dry_run": False,
            "messages": msgs, "estimated_before": 120000, "estimated_after": 90000,
            "reason": "trigger_exceeded",
        }

    monkeypatch.setattr(provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large"))

    local_calls = []

    async def fake_local(req, path, endpoint=None):
        local_calls.append(True)
        return _Resp(200)

    remote_calls = []

    async def fake_remote(req, path, cfg):
        remote_calls.append(cfg["name"])
        return _Resp(200, b"remote")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        result = await provider._proxy_with_fallback_cycle(
            _FakeRequest(json.dumps({"model": "test", "messages": _messages(40)}).encode()),
            "v1/chat/completions", _model_config(), _CFG,
        )

    assert result.status_code == 200
    assert local_calls == [], "still-oversized compaction must not dispatch local"
    assert remote_calls == ["remote-primary"]


@pytest.mark.asyncio
async def test_remote_with_guidance_preserves_remote_fallback(monkeypatch):
    """AC3: ``remote_with_guidance`` continues to the next remote provider."""
    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        return {
            "action": "remote_with_guidance", "applied": False, "dry_run": False,
            "messages": msgs, "estimated_before": 120000,
            "reason": "summarizer_unavailable",
        }

    monkeypatch.setattr(provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large"))

    local_calls = []

    async def fake_local(req, path, endpoint=None):
        local_calls.append(True)
        return _Resp(200)

    remote_calls = []

    async def fake_remote(req, path, cfg):
        remote_calls.append(cfg["name"])
        return _Resp(200, b"remote")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        result = await provider._proxy_with_fallback_cycle(
            _FakeRequest(json.dumps({"model": "test", "messages": _messages(40)}).encode()),
            "v1/chat/completions", _model_config(), _CFG,
        )

    assert result.status_code == 200
    assert local_calls == []
    assert remote_calls == ["remote-primary"]


@pytest.mark.asyncio
async def test_compaction_exception_fails_open(monkeypatch):
    """AC3: an exception in the bypass compaction helper leaves the original
    bypass behaviour intact (remote fallback, no crash)."""
    monkeypatch.setattr(provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large"))

    local_calls = []

    async def fake_local(req, path, endpoint=None):
        local_calls.append(True)
        return _Resp(200)

    remote_calls = []

    async def fake_remote(req, path, cfg):
        remote_calls.append(cfg["name"])
        return _Resp(200, b"remote")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    with patch(
        "proxy.router_helpers.evaluate_and_apply_compaction",
        side_effect=RuntimeError("boom"),
    ):
        result = await provider._proxy_with_fallback_cycle(
            _FakeRequest(json.dumps({"model": "test", "messages": _messages(40)}).encode()),
            "v1/chat/completions", _model_config(), _CFG,
        )

    assert result.status_code == 200
    assert local_calls == []
    assert remote_calls == ["remote-primary"]
