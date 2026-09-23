"""
Integration test: ``context_pressure`` advisory and compaction trigger agree.

Regression for LP-0MU5FTBOH005BR92 (parent LP-0MU5A84YU003YOTY).

Parent bug: the routing / ``context_pressure`` advisory estimated the session
with the native tokenizer, while the compaction trigger estimated it with
tiktoken.  A session whose native estimate was above the trigger (advisory
fires) could have a tiktoken estimate *below* it, so compaction decided
``noop`` / ``below_trigger`` and logged no ``compaction_event`` — the session
was told to compact but was never compacted.

These tests build a session in exactly that regime
(``tiktoken_estimate < trigger < native_estimate``) and assert:

* the routing advisory fires on the same estimate the compaction path uses;
* the compaction decision is never a silent below-trigger ``noop``;
* a ``compaction_event`` is emitted through the real ``_handle_session``
  wiring (fast and cheap modes).

The regime precondition is asserted explicitly: if the fixture ever stops
diverging between the two tokenizers the test fails loudly rather than
silently proving nothing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from proxy.compaction import (
    compaction_trigger_tokens,
    decide_session_compaction,
)
from proxy.provider import (
    _estimate_prompt_tokens_for_routing,
    _get_tokenizer_for_model,
    should_warn_context_pressure,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

try:
    from benchmarks import slot_benchmark as sb
except ImportError:  # pragma: no cover - package layout fallback
    from proxy.benchmarks import slot_benchmark as sb

# Operator-approved schedules (LP-0MTVXP7DG00613ZB / parent AC5):
FAST_CONFIG = {
    "local_model_ctx_size": 262144,
    "session_slot_pool_size": 3,
    "compaction_trigger_ratio": 0.70,
}
CHEAP_CONFIG = {
    "local_model_ctx_size": 131072,
    "session_slot_pool_size": 2,
    "compaction_trigger_ratio": 0.70,
}
NATIVE_MODEL_CONFIG = {"tokenizer": "qwen3"}

# Fixture size chosen so the native Qwen3 estimate clears the fast trigger
# (58,300) while the tiktoken estimate stays well below it — the exact
# regime that produced the advisory-without-compaction bug.
FIXTURE_SIZE = 46000


def make_large_session(size: int = FIXTURE_SIZE) -> list[dict]:
    """A compactable session whose native estimate exceeds the trigger."""
    text = sb.generate_large_prompt_fixture(size)
    return [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]


def stub_summarizer(middle_messages, previous_summary=None) -> str:
    return f"SUMMARY of {len(middle_messages)} messages"


def native_estimate(messages: list[dict], server_config: dict) -> int:
    tokenizer, multiplier = _get_tokenizer_for_model(NATIVE_MODEL_CONFIG, server_config)
    assert tokenizer is not None, "qwen3 tokenizer must load"
    assert multiplier == 1.0
    return _estimate_prompt_tokens_for_routing(
        {"messages": messages}, tokenizer=tokenizer
    )


def make_server(server_config: dict, session_messages: list[dict] | None = None):
    srv = MagicMock()
    srv.config = {"server": dict(server_config)}
    srv.logger = MagicMock()
    session = MagicMock()
    session.session_id = "sess-agreement"
    session.messages = list(session_messages or [])
    session.message_count = 0
    session.restore_confirmed = True
    srv.session_manager = MagicMock()
    srv.session_manager.get_or_create = AsyncMock(return_value=(session, True))
    srv.session_manager.compute_delta = MagicMock(return_value=([], True))
    return srv


async def capture_compaction_estimate(
    server_config: dict,
    body_json: dict,
    model_config: dict = NATIVE_MODEL_CONFIG,
) -> callable:
    """Run the real ``_handle_session`` wiring and return the production
    ``estimate_tokens`` closure handed to the compaction decision."""
    from proxy.router_helpers import _handle_session

    srv = make_server(server_config)
    captured: dict = {}

    def capture(srv_cap, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        captured["estimate_tokens"] = estimate_tokens
        return {
            "action": "noop",
            "applied": False,
            "dry_run": True,
            "messages": msgs,
            "reason": "dry_run",
        }

    with patch("proxy.lifecycle.get_model_config", return_value=model_config), patch(
        "proxy.router_helpers._evaluate_session_compaction", side_effect=capture
    ):
        await _handle_session(srv, body_json, server_config, {})

    assert captured.get("estimate_tokens") is not None
    return captured["estimate_tokens"]


# ===================================================================
# The regression regime itself
# ===================================================================


@pytest.mark.parametrize(
    "server_config",
    [FAST_CONFIG, CHEAP_CONFIG],
    ids=["fast-58300", "cheap-43008"],
)
def test_fixture_is_in_the_divergence_regime(server_config):
    """Precondition for every other test: the session's tiktoken estimate
    sits below the trigger while the native estimate clears it.  This is the
    exact window in which the old wiring produced advisory-without-compaction.
    """
    messages = make_large_session()
    trigger = compaction_trigger_tokens("fast", server_config)
    assert trigger > 0

    native = native_estimate(messages, server_config)
    tiktoken = _estimate_prompt_tokens_for_routing({"messages": messages})

    assert tiktoken < trigger < native, (
        f"fixture must straddle the trigger (tiktoken={tiktoken} < "
        f"trigger={trigger} < native={native}); otherwise the test proves "
        f"nothing about the tokenizer-mismatch bug"
    )


# ===================================================================
# AC1/AC2 — advisory fires and compaction never silently noops
# ===================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_config",
    [FAST_CONFIG, CHEAP_CONFIG],
    ids=["fast-58300", "cheap-43008"],
)
async def test_advisory_session_never_silently_noops(server_config):
    """A session that raises the ``context_pressure`` advisory must produce a
    real compaction decision — never a silent below-trigger ``noop``
    (parent AC4 / child AC1-AC2)."""
    messages = make_large_session()
    body_json = {"model": "Qwen3", "messages": list(messages)}

    # Advisory fires on the routing (native) estimate.
    routing = native_estimate(messages, server_config)
    assert should_warn_context_pressure(routing, server_config), (
        "precondition: session must trigger the context_pressure advisory"
    )

    # The compaction path consumes the estimator wired by _handle_session
    # (which now resolves the native tokenizer).
    estimate_fn = await capture_compaction_estimate(server_config, body_json)
    plan = decide_session_compaction(
        messages,
        server_config,
        "fast",
        summarizer=stub_summarizer,
        estimate_tokens=estimate_fn,
        session_id="sess-agreement",
    )

    assert not (plan["action"] == "noop" and plan["reason"] == "below_trigger"), (
        f"compaction silently nooped below trigger ({plan}) while the "
        f"context_pressure advisory fired at {routing} tokens"
    )
    # The trigger really fired and a decision was made.
    assert plan["estimated_before"] > plan["trigger_tokens"]
    assert plan["action"] in ("compact", "remote_with_guidance"), plan


# ===================================================================
# AC2 — the full _handle_session path emits a compaction_event
# ===================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_config",
    [FAST_CONFIG, CHEAP_CONFIG],
    ids=["fast-58300", "cheap-43008"],
)
async def test_full_wiring_emits_a_compaction_event(server_config):
    """Driving the real ``_handle_session`` must emit a ``compaction_event``
    for an advisory-triggering session (not silence)."""
    from proxy.router_helpers import _handle_session

    messages = make_large_session()
    body_json = {"model": "Qwen3", "messages": list(messages)}
    srv = make_server(server_config)
    events: list[dict] = []

    def capture_event(plan_result, **kwargs):
        events.append(plan_result)
        return None

    with patch("proxy.lifecycle.get_model_config", return_value=NATIVE_MODEL_CONFIG), patch(
        "proxy.compaction_summarizer.build_compact_summarizer",
        return_value=stub_summarizer,
    ), patch("proxy.compaction.log_compaction_event", side_effect=capture_event):
        await _handle_session(srv, body_json, server_config, {})

    assert events, (
        "no compaction_event emitted for an advisory-triggering session — "
        "the original silence bug"
    )
    assert events[0]["action"] in ("compact", "remote_with_guidance"), events[0]
    assert events[0]["estimated_before"] > events[0]["trigger_tokens"]
