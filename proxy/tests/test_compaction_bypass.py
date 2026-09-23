"""
Regression test: compaction evaluated on bypassed sessions.

Feature: LP-0MU5ARWSP001BYYB (server-side compaction unreachable for oversized
sessions — routing bypass runs before session handling).
Child: LP-0MU5I99TW000ALCO.

This is the end-to-end regression test for the parent ACs. Before the fix,
``_proxy_with_fallback_cycle`` ``continue``d past local dispatch (to the
remote chain) *before* any session handling, so an oversized session was never
evaluated for server-side compaction. This test drives an over-trigger,
cache-cold request through ``_proxy_with_fallback_cycle`` with all remotes
unavailable and asserts:

  (a) compaction is evaluated (would fail on the pre-fix code), and
  (b) with a working summariser the compacted history is what gets
      dispatched — while a summariser failure surfaces
      ``remote_with_guidance`` instead of silently looping.
"""

from __future__ import annotations

import json
from unittest.mock import patch

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


def _large_messages(num_turns: int = 60) -> list[dict]:
    """History above the fast-mode compaction trigger (~58K tokens)."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Initial question."},
        {"role": "assistant", "content": "Initial answer."},
    ]
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"user_question_{i}"})
        msgs.append({"role": "assistant", "content": f"assistant_answer_{i}"})
    return msgs


async def _remote_unavailable(req, path, cfg):
    return _Resp(502, b"remote unavailable")


async def _remote_ok(req, path, cfg):
    return _Resp(200, json.dumps({"choices": [{"message": {"content": "remote"}}]}).encode())


_COMPACTED = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<summary>Earlier turns folded.</summary>"},
    {"role": "user", "content": "Recent question."},
    {"role": "assistant", "content": "Recent answer."},
]


@pytest.fixture(autouse=True)
def reset_provider_state():
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


def _request() -> _FakeRequest:
    return _FakeRequest(
        json.dumps({"model": "test", "messages": _large_messages()}).encode()
    )


# ---------------------------------------------------------------------------
# AC5(a) — Compaction evaluated on bypass with all remotes unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_evaluated_when_all_remotes_unavailable(monkeypatch, caplog):
    """AC1/AC5(a): a bypassed oversized request is evaluated for compaction
    even though every remote is unavailable.

    Regression: on the pre-fix code ``_handle_session`` (and therefore
    compaction) was never reached, so ``compaction_calls`` stayed empty.
    """
    compaction_calls = []

    monkeypatch.setattr(
        provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large")
    )

    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        compaction_calls.append({"num_messages": len(msgs), "mode": mode})
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": msgs, "reason": "summarizer_unavailable",
        }

    async def fake_local(req, path, endpoint=None):
        raise AssertionError("local must not be dispatched when bypassed")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_unavailable)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            with pytest.raises(provider.ChainExhaustedError):
                await provider._proxy_with_fallback_cycle(
                    _request(), "v1/chat/completions", _model_config(), _CFG,
                )

    assert len(compaction_calls) == 1, (
        "compaction must be evaluated for a bypassed oversized session"
    )
    assert compaction_calls[0]["num_messages"] == len(_large_messages())
    assert compaction_calls[0]["mode"] == "fast"
    assert any("compaction_bypass_eval" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_compaction_evaluated_for_economic_bypass(monkeypatch):
    """AC1: ``large_context_bypass`` (economic cold-cache skip) is also
    evaluated for compaction, not just ``context_too_large``."""
    calls = []

    monkeypatch.setattr(
        provider, "_should_skip_local", lambda *a, **k: (True, "large_context_bypass")
    )

    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        calls.append(1)
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": msgs, "reason": "below_trigger",
        }

    async def fake_local(req, path, endpoint=None):
        return _Resp(200)

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_unavailable)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with pytest.raises(provider.ChainExhaustedError):
            await provider._proxy_with_fallback_cycle(
                _request(), "v1/chat/completions", _model_config(), _CFG,
            )

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# AC5(b) — Working summariser dispatches the compacted history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_working_summariser_dispatches_compacted_history(monkeypatch):
    """AC2/AC5(b): with a working summariser the compacted history is what
    gets dispatched to local — the request does not fall through to remotes
    (which are all unavailable)."""
    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        return {
            "action": "compact", "applied": True, "dry_run": False,
            "messages": list(_COMPACTED),
            "estimated_before": 120000, "estimated_after": 1500,
            "reason": "trigger_exceeded",
            "summary_text": "<summary>", "turns_summarized": 60, "recent_turns_kept": 2,
        }

    # Bypass decision first, then the post-compaction re-check passes.
    skip_returns = [(True, "context_too_large"), (False, None)]
    monkeypatch.setattr(
        provider, "_should_skip_local",
        lambda *a, **k: skip_returns.pop(0) if skip_returns else (False, None),
    )

    dispatched = []

    async def fake_local(req, path, endpoint=None):
        dispatched.append(json.loads(await req.body()))
        return _Resp(200, b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_unavailable)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        result = await provider._proxy_with_fallback_cycle(
            _request(), "v1/chat/completions", _model_config(), _CFG,
        )

    assert result.status_code == 200
    assert len(dispatched) == 1, "exactly one local dispatch"
    assert dispatched[0]["messages"] == _COMPACTED, (
        "the compacted history must be the dispatched body"
    )


# ---------------------------------------------------------------------------
# AC5(b) — Summariser failure surfaces remote_with_guidance, no silent loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summariser_failure_surfaces_remote_with_guidance(monkeypatch, caplog):
    """AC1/AC5(b): when the summariser fails the decision is
    ``remote_with_guidance`` and the request proceeds to the remote provider
    (observable), rather than being silently unevaluated or looping."""
    def fake_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        return {
            "action": "remote_with_guidance", "applied": False, "dry_run": False,
            "messages": msgs, "estimated_before": 120000,
            "reason": "summarizer_unavailable",
        }

    monkeypatch.setattr(
        provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large")
    )

    local_calls = []

    async def fake_local(req, path, endpoint=None):
        local_calls.append(True)
        return _Resp(200)

    remote_calls = []

    async def fake_remote(req, path, cfg):
        remote_calls.append(cfg["name"])
        return _Resp(200, b'{"choices":[{"message":{"content":"remote"}}]}')

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            result = await provider._proxy_with_fallback_cycle(
                _request(), "v1/chat/completions", _model_config(), _CFG,
            )

    assert result.status_code == 200
    assert local_calls == [], "remote_with_guidance must not dispatch local"
    assert remote_calls == ["remote-primary"]
    assert any("compaction_bypass_eval" in m for m in caplog.messages), (
        "the compaction outcome must be observable in logs"
    )
