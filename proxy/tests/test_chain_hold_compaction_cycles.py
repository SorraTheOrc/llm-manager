"""Chain-hold cycle semantics regression tests (LP-0MU5AIAAY003KVM0).

Proves the decided semantics (option (a)): each chain-hold cycle re-runs the
FULL fallback cycle, including local routing, session handling and compaction
evaluation. A summarizer failure in one cycle therefore does not pin the held
request for the whole hold — a later cycle re-evaluates compaction, and a
success refreshes the dispatched request body.

Also asserts the observability contract: every ``routing_check`` emission
carries ``chain_hold_cycle=N`` so repeated lines from one held request are
distinguishable from genuine new demand.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import proxy.provider as provider
import pytest


class _State:
    """Minimal stand-in for Starlette's ``request.state``."""


class _FakeRequest:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.state = _State()
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


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


def _model_config() -> dict:
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-primary", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ]
    }


_CFG = {
    "server": {
        "llama_server_port": 8080,
        "local_model_ctx_size": 262144,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 2,
    }
}


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


async def _remote_fail(_req, _path, _cfg):
    return _Resp(502, b"remote unavailable")


def _request() -> _FakeRequest:
    return _FakeRequest(
        json.dumps({"model": "test", "messages": _messages(40)}).encode()
    )


@pytest.mark.asyncio
async def test_compaction_reevaluated_every_hold_cycle(monkeypatch, caplog):
    """AC1/AC6: compaction is evaluated once per cycle for one held request."""
    monkeypatch.setattr(
        provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large")
    )
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_fail)

    eval_calls: list[int] = []

    def fake_eval(_srv, _sid, _msgs, _mode, summarizer=None, estimate_tokens=None, **_kw):
        eval_calls.append(1)
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": _msgs, "reason": "below_trigger",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            result = await provider.proxy_with_fallback(
                _request(), "v1/chat/completions", _model_config(), _CFG
            )

    assert result.status_code == 503
    # cycle 0 + chain_hold_max_cycles(2) retries = 3 evaluations.
    assert len(eval_calls) == 3, (
        f"compaction must be re-evaluated per hold cycle, got {len(eval_calls)}"
    )
    routing_lines = [m for m in caplog.messages if m.startswith("routing_check")]
    assert len(routing_lines) == 3, f"expected 3 routing_check lines, got {routing_lines}"
    cycles = sorted(int(m.split("chain_hold_cycle=")[1].split()[0]) for m in routing_lines)
    assert cycles == [0, 1, 2], f"expected cycle stamps 0,1,2, got {cycles}"


@pytest.mark.asyncio
async def test_summarizer_recovers_on_later_cycle_and_dispatches_compacted(
    monkeypatch, caplog
):
    """AC3: a summarizer failure on cycle 0 must not pin the request; a later
    cycle compacts and dispatches the compacted history."""
    compacted = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "<summary>...</summary>"},
        {"role": "assistant", "content": "recent"},
    ]
    # Call 1: bypass prunes local; call 3: post-compaction re-check returns local.
    skip_returns = [(True, "context_too_large"), (True, "context_too_large"), (False, None)]

    def fake_skip(*_a, **_k):
        return skip_returns.pop(0) if skip_returns else (False, None)

    monkeypatch.setattr(provider, "_should_skip_local", fake_skip)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_fail)

    eval_calls: list[int] = []

    def fake_eval(_srv, _sid, msgs, _mode, summarizer=None, estimate_tokens=None, **_kw):
        eval_calls.append(1)
        if len(eval_calls) == 1:
            return {
                "action": "noop", "applied": False, "dry_run": True,
                "messages": msgs, "reason": "summarizer_failed",
            }
        return {
            "action": "compact", "applied": True, "dry_run": False,
            "messages": compacted,
            "estimated_before": 120000, "estimated_after": 2000,
            "reason": "trigger_exceeded", "summary_text": "...",
            "turns_summarized": 38, "recent_turns_kept": 2,
        }

    local_bodies: list[dict] = []

    async def fake_local(req, _path, endpoint=None):
        local_bodies.append(json.loads(await req.body()))
        return _Resp(200, b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        with caplog.at_level("INFO"):
            result = await provider.proxy_with_fallback(
                _request(), "v1/chat/completions", _model_config(), _CFG
            )

    assert result.status_code == 200
    assert len(eval_calls) == 2, "summarizer failure must not stop later re-evaluation"
    assert len(local_bodies) == 1, "exactly one local dispatch after the successful compaction"
    assert local_bodies[0]["messages"] == compacted, (
        "the compacted history must be dispatched on the recovering cycle"
    )
    assert any("routing_compaction_local_redispatch" in m for m in caplog.messages)
