"""Total chain-hold budget + local first-byte ceiling bound below 900 s.

Covers the AC1 gap from the LP-0MU1S617L002Q9PR audit (2026-10-12): the
chain-hold loop could hold a request for ``chain_hold_seconds(300) *
chain_hold_max_cycles(3) = 900 s`` — exactly the client's 900-second
queue-timeout deadline — and the local dispatch first-byte ceiling
(``local_dispatch_max_prefill_seconds`` default 900) logged
``dispatch_first_byte_timeout ... ceiling=900s``.

New safety bounds (both strictly below the 900 s client deadline):

- ``server.chain_hold_total_seconds`` (default 720): a cumulative budget for
  ALL chain-hold waits across cycles (streaming + non-streaming). Once the
  budget is consumed the terminal exhaustion is returned instead of holding
  again. Zero disables the cap (legacy cycle-bounded behaviour).
- ``server.local_dispatch_first_byte_timeout_seconds`` (default 840): bounds
  the initial first-byte wait for a local dispatch so the proxy returns a
  retryable 503 before the client's 900 s deadline. The lifetime wedge bound
  (``local_dispatch_max_prefill_seconds``) is unchanged.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import proxy.provider as provider
import pytest
from fastapi import Response

CLIENT_QUEUE_TIMEOUT = 900.0


class _DummyRequest:
    """Minimal request stub (mirrors test_chain_hold.py)."""

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


@pytest.fixture(autouse=True)
def reset_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield



def _two_provider_config():
    return {
        "providers": [
            {
                "name": "remote-primary",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.org/v1",
            },
        ],
        "aliases": ["test*"],
    }


async def _collect(result) -> str:
    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    return "".join(chunks)


# ===================================================================
# AC1 — chain_hold_total_seconds config + defaults
# ===================================================================


def test_chain_hold_total_budget_default_is_below_client_deadline():
    """Default total hold budget is 720 s and strictly below the client's
    900 s queue-timeout deadline."""
    assert provider._get_chain_hold_total_seconds({}) == 720.0
    assert provider._get_chain_hold_total_seconds({}) < CLIENT_QUEUE_TIMEOUT


def test_chain_hold_total_budget_flat_and_server_overrides():
    assert provider._get_chain_hold_total_seconds({"chain_hold_total_seconds": 500}) == 500.0
    assert provider._get_chain_hold_total_seconds({"server": {"chain_hold_total_seconds": 400}}) == 400.0
    # Missing knob falls back to the documented default even when other knobs
    # are present.
    assert provider._get_chain_hold_total_seconds(
        {"server": {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3}}
    ) == 720.0


def test_validate_chain_hold_total_budget():
    """Validation rejects negatives, non-numbers and budgets that sit at or
    above the client's 900 s deadline; zero (opt-out) and sane values pass."""
    assert provider.validate_chain_hold_config(
        {"server": {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3,
                    "chain_hold_total_seconds": 720}}
    ) == []

    problems = provider.validate_chain_hold_config(
        {"server": {"chain_hold_total_seconds": -1}}
    )
    assert any("chain_hold_total_seconds" in p and ">= 0" in p for p in problems)

    problems = provider.validate_chain_hold_config(
        {"chain_hold_total_seconds": "abc"}
    )
    assert any("chain_hold_total_seconds" in p and "must be a number" in p for p in problems)

    problems = provider.validate_chain_hold_config(
        {"server": {"chain_hold_total_seconds": 900}}
    )
    assert any("900" in p and "chain_hold_total_seconds" in p for p in problems)
    problems = provider.validate_chain_hold_config(
        {"server": {"chain_hold_total_seconds": 1800}}
    )
    assert any("900" in p and "chain_hold_total_seconds" in p for p in problems)

    # 0 = opt out of the cap (legacy cycle-bounded behaviour) — no problem.
    assert provider.validate_chain_hold_config(
        {"server": {"chain_hold_seconds": 300, "chain_hold_max_cycles": 3,
                    "chain_hold_total_seconds": 0}}
    ) == []


def _exhausting_cycle_fn(counter: list[int]):
    """A chain cycle that always raises ChainExhaustedError (503, legacy no
    Retry-After so ``_bounded_hold_seconds`` keeps the full hold)."""

    async def _fn(_req, _path, _model, _config):
        counter.append(1)
        raise provider.ChainExhaustedError(
            Response(status_code=503, content=b"all exhausted")
        )

    return _fn


# ===================================================================
# AC2 — cumulative holds never exceed the total budget
# ===================================================================


@pytest.mark.asyncio
async def test_total_budget_bounds_cumulative_holds_non_streaming():
    """Non-streaming: with a tiny total budget the request exhausts after the
    budget is consumed, instead of running all chain_hold_max_cycles holds.

    hold=0.05, max_cycles=10, total=0.15: cycle 0 + holds after cycles
    0/1/2 consume the budget, so cycle 3 returns the exhaustion without
    holding — exactly 4 chain executions (not 11)."""
    request = _DummyRequest()
    cfg = {
        "chain_hold_seconds": 0.05,
        "chain_hold_max_cycles": 10,
        "chain_hold_total_seconds": 0.15,
    }
    counter: list[int] = []
    start = time.monotonic()
    result = await provider._run_chain_cycles(
        request, "v1/chat/completions", _two_provider_config(), cfg,
        _exhausting_cycle_fn(counter),
    )
    elapsed = time.monotonic() - start

    assert result.status_code == 503
    assert len(counter) == 4, f"expected 4 cycles, got {len(counter)}"
    # Wall time stays within the tiny budget (holds are bounded, not 10x).
    assert elapsed < 0.2, f"hold loop overshot the tiny budget: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_total_budget_bounds_cumulative_holds_streaming():
    """Streaming: the hold-feedback generator respects the same total budget
    and emits the terminal exhaustion chunk once the budget is spent.

    Cycle 0 runs in ``_run_chain_cycles`` (hold 0.05 consumed); the generator
    then holds once per remaining cycle and terminates when the budget is
    spent — 3 chain executions total, never the 11 max_cycles allow."""
    request = _DummyRequest(stream=True)
    cfg = {
        "chain_hold_seconds": 0.05,
        "chain_hold_max_cycles": 10,
        "chain_hold_total_seconds": 0.15,
    }
    counter: list[int] = []
    start = time.monotonic()
    result = await provider._run_chain_cycles(
        request, "v1/chat/completions", _two_provider_config(), cfg,
        _exhausting_cycle_fn(counter),
    )
    body = await _collect(result)
    elapsed = time.monotonic() - start

    assert "data:" in body  # terminal streamed error chunk body
    assert len(counter) == 3, f"expected 3 cycles, got {len(counter)}"
    assert elapsed < 0.2, f"streaming hold loop overshot the budget: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_total_budget_zero_disables_cap():
    """Zero total budget opts out: the max_cycles bound alone governs the
    holds (legacy behaviour: 3 holds + the initial cycle = 4 executions)."""
    request = _DummyRequest()
    cfg = {
        "chain_hold_seconds": 0.005,
        "chain_hold_max_cycles": 3,
        "chain_hold_total_seconds": 0,
    }
    counter: list[int] = []
    result = await provider._run_chain_cycles(
        request, "v1/chat/completions", _two_provider_config(), cfg,
        _exhausting_cycle_fn(counter),
    )

    assert result.status_code == 503
    assert len(counter) == 4, f"expected 4 cycles (legacy), got {len(counter)}"


# ===================================================================
# AC3 — local dispatch first-byte ceiling below the client deadline
# ===================================================================


def _make_srv(server_cfg_overrides: dict | None = None) -> SimpleNamespace:
    cfg = {
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
    }
    if server_cfg_overrides:
        cfg.update(server_cfg_overrides)
    return SimpleNamespace(config={"server": cfg})


def test_first_byte_timeout_default_below_client_deadline():
    from proxy.router_helpers import (
        _get_first_byte_timeout_seconds,
        _get_prefill_deadline_seconds,
    )

    srv = _make_srv()
    # The effective ceiling is the tighter of the wedge bound (840) and the
    # prefill deadline (90) — LP-0MVIQ0G2M0006QF6 AC3: the safety detour is
    # the bound that actually fires. The wedge bound is unchanged.
    assert _get_prefill_deadline_seconds(srv) == 90.0
    timeout = _get_first_byte_timeout_seconds(srv)
    assert timeout == 90.0
    assert timeout == _get_prefill_deadline_seconds(srv)
    assert timeout < CLIENT_QUEUE_TIMEOUT


def test_first_byte_timeout_respects_override_and_zero():
    from proxy.router_helpers import _get_first_byte_timeout_seconds

    # A wedge ceiling tighter than the deadline wins.
    srv = _make_srv({"local_dispatch_first_byte_timeout_seconds": 30})
    assert _get_first_byte_timeout_seconds(srv) == 30.0

    # A looser one is tightened to the deadline (90).
    srv = _make_srv({"local_dispatch_first_byte_timeout_seconds": 600})
    assert _get_first_byte_timeout_seconds(srv) == 90.0

    # Disabling the wedge ceiling must not disable the deadline backstop.
    srv = _make_srv({"local_dispatch_first_byte_timeout_seconds": 0})
    assert _get_first_byte_timeout_seconds(srv) == 90.0

    # Disabling BOTH restores the previous "no ceiling" behaviour (0).
    srv = _make_srv(
        {
            "local_dispatch_first_byte_timeout_seconds": 0,
            "local_prefill_deadline_seconds": 0,
        }
    )
    assert _get_first_byte_timeout_seconds(srv) == 0.0

    # The legacy default is still honoured when the deadline is disabled.
    srv = _make_srv({"local_prefill_deadline_seconds": 0})
    assert _get_first_byte_timeout_seconds(srv) == 840.0
