"""Per-turn correlation id (``turn=``) on structured proxy log lines.

LP-0MVMBM467005BT4A. The id is generated once per client request and carried
on every structured line the turn emits, so a request's lifecycle
(``routing_skip_local`` -> ``Fallback triggered`` -> ``Stream started`` ->
``Stream finished``) is reconstructable exactly instead of by FIFO/nearest-line
heuristics.

These tests assert observable behaviour: the id in emitted log text, its
stability across fallback hops, its uniqueness between concurrent turns, and
the backwards-compatible absence of the field when no turn is bound.
"""

import asyncio
import contextvars
import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import proxy.provider as provider
import pytest
from fastapi import Response
from proxy.turn_id import (
    TURN_ID_LENGTH,
    bind_turn_id,
    get_turn_id,
    new_turn_id,
    reset_turn_id,
    turn_field,
)

# Not a logging call: plain pattern construction, so no lazy-formatting
# semantics are involved in using an f-string here.
TURN_RE = re.compile(rf"\bturn=([0-9a-f]{{{TURN_ID_LENGTH}}})\b")


class _DummyRequest:
    """Minimal ASGI-ish request stub (mirrors the fallback-test helper)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.state = SimpleNamespace()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def _fresh_turn():
    """Start every test with no turn bound in this context."""
    reset_turn_id()
    yield
    reset_turn_id()


@pytest.fixture(autouse=True)
def reset_provider_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider._last_routing_hop.clear()
    yield


@pytest.fixture(autouse=True)
def pin_priority(monkeypatch):
    monkeypatch.setattr(
        "proxy.provider.request_priority", lambda *a, **k: "medium"
    )


@pytest.fixture
def mixed_model_config():
    return {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _turn_of(message: str) -> str | None:
    m = TURN_RE.search(message)
    return m.group(1) if m else None


def _fallback_line(caplog) -> str:
    for record in reversed(caplog.records):
        if "Fallback triggered" in record.getMessage():
            return record.getMessage()
    raise AssertionError("no 'Fallback triggered' log record emitted")


# ---------------------------------------------------------------------------
# Helper semantics
# ---------------------------------------------------------------------------


def test_new_turn_id_is_short_hex():
    turn = new_turn_id()

    assert re.fullmatch(rf"[0-9a-f]{{{TURN_ID_LENGTH}}}", turn), turn


def test_new_turn_ids_differ():
    ids = {new_turn_id() for _ in range(200)}

    assert len(ids) == 200


def test_turn_field_is_empty_when_no_turn_is_bound():
    assert get_turn_id() is None
    assert turn_field() == ""


def test_bind_turn_id_is_idempotent_within_one_turn():
    first = bind_turn_id()

    second = bind_turn_id()

    assert first == second
    assert turn_field() == f" turn={first}"


def test_bind_turn_id_mirrors_onto_request_state():
    request = _DummyRequest()

    turn = bind_turn_id(request)

    assert request.state.turn_id == turn
    assert get_turn_id(request) == turn


def test_request_state_wins_when_context_lost():
    """A line logged from a task that lost the context still attributes."""
    request = _DummyRequest()
    turn = bind_turn_id(request)
    reset_turn_id()  # simulate a detached task (e.g. the disconnect reaper)

    assert get_turn_id(request) == turn
    assert turn_field(request) == f" turn={turn}"


def test_concurrent_turns_get_distinct_ids():
    """Two overlapping requests must never share an id (AC2 uniqueness)."""

    async def _one_turn():
        await asyncio.sleep(0)
        turn = bind_turn_id()
        await asyncio.sleep(0)
        return turn, get_turn_id()

    async def _run():
        return await asyncio.gather(_one_turn(), _one_turn(), _one_turn())

    results = asyncio.run(_run())
    ids = [turn for turn, _ in results]

    assert len(set(ids)) == 3, results
    # Each turn keeps its own id across the await points.
    assert all(seen == issued for issued, seen in results), results


# ---------------------------------------------------------------------------
# Emit sites carry the id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_chain_carries_one_stable_turn_across_the_hop(
    mixed_model_config, caplog
):
    """The fallback line's id equals the id seen by each dispatch hop (AC1/AC2)."""
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    request = _DummyRequest()
    seen: list[str | None] = []

    async def _mock_proxy_to_local(_req, _path):
        seen.append(get_turn_id())
        raise httpx.ConnectError("Connection refused to llama-server")

    async def _mock_proxy_to_remote(_req, _path, _cfg):
        seen.append(get_turn_id())
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config,
            {"provider_cooldown_seconds": 60},
        )

    assert result.status_code == 200
    turn = _turn_of(_fallback_line(caplog))
    assert turn, _fallback_line(caplog)
    # Stable across the local->remote hop: every hop saw the same id.
    assert seen == [turn, turn], seen


@pytest.mark.asyncio
async def test_routing_hop_line_carries_turn(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    turn = bind_turn_id()

    provider.note_routing_hop("sess-turn-hop", "remote", model_name="m")

    lines = [r.getMessage() for r in caplog.records if "routing_hop" in r.getMessage()]
    assert lines, [r.getMessage() for r in caplog.records]
    assert _turn_of(lines[0]) == turn


def test_provider_context_skip_line_carries_turn(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    turn = bind_turn_id()

    provider._record_provider_context_skip(
        [], "remote-x", "remote", 200000, 150000, "sess-ctx-skip", "default"
    )

    line = next(
        r.getMessage()
        for r in caplog.records
        if "routing_skip_provider_context" in r.getMessage()
    )
    assert _turn_of(line) == turn


def test_log_request_line_carries_turn(caplog):
    """``[local] POST`` / ``[remote] POST`` carry the id (AC1)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import log_request

    request = _DummyRequest(body=b'{"messages":[]}')

    log_request(request, b'{"messages":[]}', "local", session_id="sess-log-req")

    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("[local] POST")
    )
    turn = _turn_of(line)
    assert turn == request.state.turn_id
    assert turn == get_turn_id()


def test_log_request_binds_a_turn_when_none_is_bound(caplog):
    """A direct call still attributes its line (no turn => generated)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import log_request

    request = _DummyRequest()
    assert get_turn_id() is None

    log_request(request, b'{"messages":[]}', "local")

    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("[local] POST")
    )
    assert _turn_of(line) == request.state.turn_id


def test_stream_finished_line_carries_turn(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import log_response_chunk

    turn = bind_turn_id()
    chunk = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'

    log_response_chunk(chunk, session_id="sess-stream", model="m", provider="local")

    line = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Stream finished:")
    )
    assert _turn_of(line) == turn


def test_stream_finished_line_omits_turn_when_unbound(caplog):
    """Legacy compatibility: no turn bound => the pre-change line shape (AC4)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import log_response_chunk

    chunk = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'

    log_response_chunk(chunk, session_id="sess-legacy", model="m", provider="local")

    line = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Stream finished:")
    )
    assert "turn=" not in line
    assert line.startswith("Stream finished: reason=stop"), line
    assert "session=sess-legacy" in line, line
    assert "provider=local model=m" in line, line


def test_deferred_stream_finish_is_keyed_by_turn(caplog):
    """Two interleaved turns cannot complete each other's deferred line (AC2)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import (
        flush_pending_stream_finish,
        log_response_chunk,
    )

    finish_chunk = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
    usage_chunk = (
        b'data: {"choices":[], "usage":{"prompt_tokens":7,"completion_tokens":1,'
        b'"total_tokens":8}}\n\n'
    )

    turn_a = bind_turn_id()
    ctx_a = contextvars.copy_context()
    # Turn A defers its terminal line, waiting for its own usage chunk.
    log_response_chunk(
        finish_chunk, session_id="sess-parallel", model="m", provider="local",
        entry="e", expect_usage=True,
    )

    reset_turn_id()
    turn_b = bind_turn_id()
    # Turn B's usage chunk must NOT complete turn A's deferred line.
    log_response_chunk(
        usage_chunk, session_id="sess-parallel", model="m", provider="local",
        entry="e", expect_usage=True,
    )

    def _finished_lines():
        return [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Stream finished:")
        ]

    assert turn_a != turn_b
    assert _finished_lines() == []
    assert (
        flush_pending_stream_finish(
            session_id="sess-parallel", model="m", provider="local", entry="e"
        )
        is False
    )
    assert ctx_a.run(
        flush_pending_stream_finish,
        session_id="sess-parallel",
        model="m",
        provider="local",
        entry="e",
    ) is True
    lines = _finished_lines()
    assert len(lines) == 1
    assert _turn_of(lines[0]) == turn_a


def test_non_stream_usage_line_carries_turn(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.router_helpers import log_non_stream_usage

    turn = bind_turn_id()
    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()

    log_non_stream_usage(body, session_id="sess-nonstream", model="m", provider="local")

    line = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Stream finished:")
    )
    assert _turn_of(line) == turn
    assert "stream=false" in line
