"""
Tests for chain-hold retry (LP-0MSH94Z7K007VKC9).

When every provider in a model's fallback chain is exhausted, the proxy holds
the request for ``server.chain_hold_seconds`` (default 300) instead of
erroring immediately, then starts a NEW cycle from the FIRST provider. The
number of hold-retry cycles is bounded by ``server.chain_hold_max_cycles``
(default 3; 0 = infinite).

Acceptance criteria covered:
- AC1 — Hold then new cycle: cycle 0 exhausts -> response succeeds on cycle 2.
- AC2 — Bounded cycles: error surfaces after exactly max_cycles hold-retry
  cycles; 0 keeps retrying until disconnect.
- AC3 — SSE feedback comments (streaming): comment lines appear in the stream
  before the final content.
- AC4 — Client disconnect aborts: the hold aborts promptly.
- AC5 — Config & validation: defaults and non-negative validation.
- Regression — successful responses are unchanged; the feature is disabled
  (single-pass) when the config knobs are absent.
"""

import json
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse


class _DummyRequest:
    """Minimal request stub with optional client-disconnect simulation.

    ``disconnected_after=N`` makes ``is_disconnected()`` return True on the
    (N+1)-th call; ``None`` (default) never disconnects.
    """

    def __init__(self, body: bytes = b'{"model":"test"}', disconnected_after=None):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self._disconnect_calls = 0
        self._disconnected_after = disconnected_after

    async def body(self):
        return self._body

    async def is_disconnected(self):
        self._disconnect_calls += 1
        if self._disconnected_after is not None and self._disconnect_calls > self._disconnected_after:
            return True
        return False


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset cooldown and failure-count state between tests to avoid cross-test leakage."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


@pytest.fixture
def two_provider_config():
    """A model config with an ordered providers list (remote only)."""
    return {
        "providers": [
            {
                "name": "remote-primary",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.anthropic.com/v1",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        ],
        "aliases": ["test*"],
    }


def _fail_then_succeed_mock(fail_count: int, success_body: dict | None = None):
    """Return a proxy_to_remote mock that fails the first ``fail_count`` calls
    (HTTP 502) then returns a 200 JSON success."""
    if success_body is None:
        success_body = {"choices": [{"message": {"content": "ok"}}]}
    call_count = 0

    async def _mock(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            return Response(status_code=502, content=b"Bad gateway")
        return Response(
            content=json.dumps(success_body),
            status_code=200,
            media_type="application/json",
        )

    return _mock


def _always_fail_mock(counter: list | None = None):
    """Return a proxy_to_remote mock that always fails with HTTP 502."""
    if counter is None:
        counter = []

    async def _mock(_req, _path, provider_cfg):
        counter.append(1)
        return Response(status_code=502, content=b"Bad gateway")

    return _mock


async def _collect(result: Response) -> str:
    """Collect a StreamingResponse body into a decoded string."""
    body = b"".join([c async for c in result.body_iterator])
    return body.decode("utf-8", errors="replace")


# ===================================================================
# AC1 — Hold then new cycle
# ===================================================================


@pytest.mark.asyncio
async def test_hold_then_new_cycle_succeeds(two_provider_config):
    """AC1: cycle 0 exhausts -> request is held -> cycle 1 succeeds from the
    first provider (non-streaming)."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,  # cooldowns expire instantly so cycle 1 can retry
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 3,
    }

    with patch(
        "proxy.provider._get_proxy_to_remote",
        return_value=_fail_then_succeed_mock(fail_count=2),
    ):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 200
    body = json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_hold_then_new_cycle_mixed_chain(two_provider_config):
    """AC1 via proxy_with_fallback: local 4xx + remote fail -> hold -> remote
    succeeds on the new cycle."""
    model_config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ]
    }
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 3,
    }

    async def _mock_proxy_to_local(_req, _path):
        return Response(status_code=400, content=b"bad request")

    remote_calls = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        remote_calls.append(1)
        if len(remote_calls) == 1:
            return Response(status_code=502, content=b"Bad gateway")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.provider._get_proxy_to_local", return_value=_mock_proxy_to_local):
        with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
            result = await provider.proxy_with_fallback(
                _DummyRequest(), "v1/chat/completions", model_config, cfg
            )

    assert result.status_code == 200
    assert len(remote_calls) == 2  # cycle 0 failed, cycle 1 succeeded


# ===================================================================
# AC2 — Bounded cycles
# ===================================================================


@pytest.mark.asyncio
async def test_bounded_cycles_error_after_bound(two_provider_config):
    """AC2: with max_cycles=2 the exhaustion error surfaces after exactly 2
    hold-retry cycles (3 chain runs), not immediately."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 2,
    }

    calls = []
    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock(calls)):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 503
    # 3 chain runs (cycles 0, 1, 2) x 2 providers
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_zero_max_cycles_retries_until_disconnect(two_provider_config):
    """AC2: max_cycles=0 (infinite) keeps retrying until the client disconnects."""
    request = _DummyRequest(disconnected_after=2)  # disconnect on the 3rd hold
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 0,
    }

    calls = []
    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock(calls)):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 503
    # Cycle 0 + 2 completed holds; the 3rd hold aborts on disconnect.
    assert len(calls) == 6


# ===================================================================
# AC3 — SSE feedback comments (streaming)
# ===================================================================


@pytest.mark.asyncio
async def test_streaming_hold_emits_comment_then_success(two_provider_config):
    """AC3: a streaming hold emits the ``: chain exhausted ...`` comment line
    before the final content streams."""
    request = _DummyRequest(body=b'{"model":"test","stream":true}')
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 3,
    }

    call_count = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_count.append(1)
        if len(call_count) <= 2:
            return Response(status_code=502, content=b"Bad gateway")

        async def _body():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'hello'}, 'index': 0}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), media_type="text/event-stream")

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )
        # The streaming hold generator runs cycles lazily during body
        # consumption — keep the mock patch active while collecting.
        collected = await _collect(result)

    assert isinstance(result, StreamingResponse)
    comment_idx = collected.find(": chain exhausted")
    content_idx = collected.find("hello")
    assert comment_idx != -1, f"expected hold comment in stream: {collected!r}"
    assert content_idx != -1, f"expected content after hold: {collected!r}"
    assert comment_idx < content_idx, "comment must appear before the final content"
    # The comment names the provider the new cycle restarts from.
    assert "retrying from remote-primary" in collected


@pytest.mark.asyncio
async def test_streaming_bounded_cycles_terminal_error_chunk(two_provider_config):
    """AC2/AC3: with max_cycles=2 the stream emits 2 hold comments then the
    terminal exhaustion body as an SSE data chunk."""
    request = _DummyRequest(body=b'{"model":"test","stream":true}')
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 2,
    }

    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock()):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )
        # Keep the mock patch active while the lazy generator runs cycles.
        collected = await _collect(result)

    assert collected.count(": chain exhausted") == 2
    assert "All providers exhausted" in collected


# ===================================================================
# AC4 — Client disconnect aborts the hold
# ===================================================================


@pytest.mark.asyncio
async def test_disconnect_aborts_hold_non_streaming(two_provider_config):
    """AC4: a non-streaming request that disconnects during the hold aborts
    promptly — no new cycle runs and no wasted waiting."""
    request = _DummyRequest(disconnected_after=0)  # disconnected on first check
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 300,  # would be a long wait if not aborted
        "chain_hold_max_cycles": 3,
    }

    calls = []
    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock(calls)):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 503
    assert len(calls) == 2  # only cycle 0 ran; the hold aborted before cycle 1


@pytest.mark.asyncio
async def test_disconnect_aborts_hold_streaming(two_provider_config):
    """AC4: a streaming request that disconnects ends the hold stream promptly
    — no comments, no retry."""
    request = _DummyRequest(body=b'{"model":"test","stream":true}', disconnected_after=0)
    cfg = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 300,
        "chain_hold_max_cycles": 3,
    }

    calls = []
    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock(calls)):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )
        # Keep the mock patch active while the lazy generator runs cycles.
        collected = await _collect(result)

    assert collected == ""  # hold aborted before any comment/retry
    assert len(calls) == 2  # only cycle 0 ran


# ===================================================================
# AC5 — Config getters, enablement and validation
# ===================================================================


def test_chain_hold_disabled_when_not_configured():
    """The hold feature is disabled when neither knob is present — legacy
    single-pass behavior."""
    assert provider._chain_hold_enabled({"provider_cooldown_seconds": 60}) is False
    assert provider._chain_hold_enabled({"server": {"provider_cooldown_seconds": 60}}) is False


def test_chain_hold_enabled_when_configured():
    """Either knob (flat or server.*) enables the feature."""
    assert provider._chain_hold_enabled({"chain_hold_seconds": 300}) is True
    assert provider._chain_hold_enabled({"server": {"chain_hold_max_cycles": 3}}) is True


def test_chain_hold_defaults():
    """Defaults: 300s hold / 3 cycles when only one knob is configured."""
    assert provider._get_chain_hold_seconds({"server": {"chain_hold_max_cycles": 5}}) == 300.0
    assert provider._get_chain_hold_max_cycles({"chain_hold_seconds": 60}) == 3
    assert provider._get_chain_hold_seconds(
        {"server": {"chain_hold_seconds": 0.01, "chain_hold_max_cycles": 2}}
    ) == 0.01
    assert provider._get_chain_hold_max_cycles(
        {"server": {"chain_hold_seconds": 0.01, "chain_hold_max_cycles": 2}}
    ) == 2


def test_validate_chain_hold_config():
    """AC5: non-negative validation for chain_hold_seconds / max_cycles."""
    assert provider.validate_chain_hold_config(
        {"server": {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3}}
    ) == []

    problems = provider.validate_chain_hold_config({"server": {"chain_hold_seconds": -1}})
    assert any("chain_hold_seconds" in p and ">= 0" in p for p in problems)

    problems = provider.validate_chain_hold_config({"chain_hold_max_cycles": -2})
    assert any("chain_hold_max_cycles" in p and ">= 0" in p for p in problems)

    problems = provider.validate_chain_hold_config({"server": {"chain_hold_seconds": "abc"}})
    assert any("must be a number" in p for p in problems)

    problems = provider.validate_chain_hold_config({"chain_hold_max_cycles": "xyz"})
    assert any("must be an integer" in p for p in problems)

    # Unbounded busy-retry loop: 0 hold interval + infinite cycles.
    problems = provider.validate_chain_hold_config(
        {"chain_hold_seconds": 0, "chain_hold_max_cycles": 0}
    )
    assert any("unbounded retry loop" in p for p in problems)


def test_load_config_rejects_invalid_chain_hold(tmp_path):
    """AC5: load_config raises on invalid chain-hold config."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("server:\n  chain_hold_seconds: -5\n")

    from proxy.utils import load_config

    with pytest.raises(ValueError, match="Chain-hold config validation failed"):
        load_config(str(cfg_path))


def test_load_config_accepts_valid_chain_hold(tmp_path):
    """AC5: load_config accepts the documented defaults."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("server:\n  chain_hold_seconds: 300\n  chain_hold_max_cycles: 3\n")

    from proxy.utils import load_config

    cfg = load_config(str(cfg_path))
    assert cfg["server"]["chain_hold_seconds"] == 300
    assert cfg["server"]["chain_hold_max_cycles"] == 3


# ===================================================================
# Regression — successful responses unchanged / feature disabled
# ===================================================================


@pytest.mark.asyncio
async def test_success_unaffected_by_hold_config(two_provider_config):
    """The hold only defers the exhaustion verdict — successful responses
    (status, headers) are returned unchanged."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 3,
    }

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.provider._get_proxy_to_remote", return_value=_mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "remote-primary"


@pytest.mark.asyncio
async def test_hold_disabled_single_pass(two_provider_config):
    """No chain-hold config -> single pass; the exhaustion 503 is returned
    immediately with no hold/retry (legacy behavior preserved)."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    calls = []
    with patch("proxy.provider._get_proxy_to_remote", return_value=_always_fail_mock(calls)):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", two_provider_config, cfg
        )

    assert result.status_code == 503
    assert len(calls) == 2  # single cycle, no hold/retry
