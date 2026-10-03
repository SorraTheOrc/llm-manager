"""Chain-hold duration bound by the real ``retry_after`` (LP-0MU56ZKQD005SX08).

Covers:

- AC1: the chain-hold never waits longer than the computed ``retry_after``.
- AC2: a large ``retry_after`` (window edge / usage-limit reset) returns the
  exhaustion response without exhausting ``chain_hold_max_cycles``.
- AC3: streaming clients still receive ``: chain exhausted`` feedback during
  any hold.
- AC4: hold duration tracks small / medium / large ``retry_after`` values.
"""

from __future__ import annotations

import json

import proxy.provider as provider
import pytest
from fastapi import Response


class _Req:
    def __init__(self, body: bytes = b'{"model":"test"}', stream: bool = False):
        if stream:
            body = b'{"model":"test","stream":true}'
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _exhaustion(retry_after: int | None):
    """Build a 503 exhaustion response with (or without) a retry_after."""
    if retry_after is None:
        return Response(content=b"plain boom", status_code=503, media_type="text/plain")
    payload = {"error": "All providers exhausted", "retry_after": retry_after}
    return Response(
        content=json.dumps(payload).encode(),
        status_code=503,
        media_type="application/json",
        headers={"Retry-After": str(retry_after)},
    )


def _cycle_fn(response_factory, succeed_on: int = 99):
    calls: list[int] = []

    async def _fn(_req, _path, _model, _config):
        calls.append(1)
        if len(calls) < succeed_on:
            raise provider.ChainExhaustedError(response_factory())
        return Response(status_code=200, content=b"ok")

    return _fn, calls


async def _collect_stream(result) -> str:
    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    return "".join(chunks)


@pytest.fixture(autouse=True)
def reset_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


class TestBoundedHold:
    @pytest.mark.asyncio
    async def test_small_retry_after_bounds_hold(self, monkeypatch):
        cycle_fn, calls = _cycle_fn(lambda: _exhaustion(2), succeed_on=2)
        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 200
        assert holds == [2.0]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_medium_retry_after_caps_at_config_hold(self, monkeypatch):
        cycle_fn, _calls = _cycle_fn(lambda: _exhaustion(250), succeed_on=2)
        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 200
        assert holds == [250.0]

    @pytest.mark.asyncio
    async def test_large_retry_after_returns_immediately(self, monkeypatch):
        cycle_fn, calls = _cycle_fn(lambda: _exhaustion(36000))
        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 503
        assert holds == [], "a large retry_after must not hold"
        assert len(calls) == 1, "no further cycles after a doomed hold decision"
        assert result.headers.get("Retry-After") == "36000"

    @pytest.mark.asyncio
    async def test_zero_retry_after_retries_without_waiting(self, monkeypatch):
        cycle_fn, calls = _cycle_fn(lambda: _exhaustion(0), succeed_on=2)
        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 200
        assert holds == [0.0]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_unknown_retry_after_preserves_legacy_hold(self, monkeypatch):
        cycle_fn, _calls = _cycle_fn(lambda: _exhaustion(None), succeed_on=2)
        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 200
        assert holds == [300.0]

    @pytest.mark.asyncio
    async def test_multiple_bound_holds_never_exceed_retry_after(self, monkeypatch):
        """A medium retry_after is drained across cycles, each hold bounded."""
        responses = [_exhaustion(400), _exhaustion(100)]

        async def cycle_fn(_req, _path, _model, _config):
            if responses:
                raise provider.ChainExhaustedError(responses.pop(0))
            return Response(status_code=200, content=b"ok")

        holds: list[float] = []

        async def fake_sleep(_req, seconds):
            holds.append(seconds)
            return False

        monkeypatch.setattr(provider, "_hold_sleep", fake_sleep)
        result = await provider._run_chain_cycles(
            _Req(), "v1/chat/completions", {},
            {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
            cycle_fn,
        )

        assert result.status_code == 200
        # 400 <= 300*3 budget; first hold capped at 300, then the second
        # exhaustion's retry_after (100) bounds the next hold.
        assert holds == [300.0, 100.0]


@pytest.mark.asyncio
async def test_streaming_hold_still_emits_feedback(monkeypatch):
    """AC3: a bounded streaming hold still emits ``: chain exhausted``."""
    cycle_fn, _calls = _cycle_fn(lambda: _exhaustion(1), succeed_on=2)

    result = await provider._run_chain_cycles(
        _Req(stream=True), "v1/chat/completions", {},
        {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3},
        cycle_fn,
    )

    collected = await _collect_stream(result)
    assert ": chain exhausted" in collected
