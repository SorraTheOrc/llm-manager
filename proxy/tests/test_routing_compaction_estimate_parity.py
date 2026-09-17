"""
Regression test: routing and compaction token estimates agree.

Regression for LP-0MU5FSN3M006KKMO (parent LP-0MU5A84YU003YOTY).

Before the fix, the routing path resolved the native tokenizer via
``_get_tokenizer_for_model`` (multiplier forced to 1.0), while the
compaction trigger's estimate closure fell back to tiktoken.  The two
paths therefore measured the same session with different estimators,
so the proxy could log a ``context_pressure`` advisory (ratio >= trigger)
and then decide ``noop`` because its own compaction trigger was not
crossed.

These tests pin the parity contract:

* the compaction estimate closure resolves the *same* tokenizer the
  routing path uses (``_get_tokenizer_for_model``);
* for the same produced history the two paths return identical counts;
* routing (``max(body, session)``) is never smaller than the compaction
  estimate over the same produced history.

The tests drive the real ``_handle_session`` wiring, capturing the
``estimate_tokens`` closure that is handed to ``decide_session_compaction``
rather than re-deriving the estimator in the test (which would re-implement
production logic and prove nothing).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.provider import (
    _estimate_effective_prompt_tokens_for_routing,
    _estimate_prompt_tokens_for_routing,
    _get_tokenizer_for_model,
)

# ===================================================================
# Fixtures / helpers
# ===================================================================

SERVER_CONFIG = {
    "local_model_ctx_size": 262144,
    "session_slot_pool_size": 3,
    "compaction_trigger_ratio": 0.70,
}

NATIVE_MODEL_CONFIG = {"tokenizer": "qwen3"}
TIKTOKEN_MODEL_CONFIG = {}


def make_messages(n_turns: int = 14) -> list[dict]:
    """Build a deterministic session with known content.

    ``n_turns`` >= 12 is required for the native Qwen3 tokenizer to diverge
    from tiktoken on this content (the parity tests assert that divergence
    explicitly so a fixture that happens to tokenize identically can never
    silently weaken them).
    """
    messages: list[dict] = [{"role": "system", "content": "SYSTEM " * 40}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"user question {i} " * 60})
        messages.append({"role": "assistant", "content": f"assistant answer {i} " * 60})
    return messages


def make_session(messages: list[dict], session_id: str = "sess-parity"):
    session = MagicMock()
    session.session_id = session_id
    session.messages = list(messages)
    session.message_count = len(messages)
    session.restore_confirmed = True
    return session


def make_server(session, session_created: bool = False):
    srv = MagicMock()
    srv.config = {"server": dict(SERVER_CONFIG)}
    srv.logger = MagicMock()
    srv.session_manager = MagicMock()
    srv.session_manager.get_or_create = AsyncMock(return_value=(session, session_created))
    # Non-delta request: history matches with zero new messages.
    srv.session_manager.compute_delta = MagicMock(return_value=([], True))
    return srv


def make_request(session_id: str = "sess-parity"):
    request = MagicMock()
    request.headers = {"x-session-id": session_id}
    return request


async def capture_compaction_estimate(
    session,
    body_json: dict,
    server_config: dict,
    model_config: dict,
) -> "callable":
    """Run ``_handle_session`` and return the captured ``estimate_tokens``
    closure handed to ``_evaluate_session_compaction``."""
    from proxy.router_helpers import _handle_session

    srv = make_server(session)
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

    assert captured.get("estimate_tokens") is not None, (
        "compaction wiring must pass an estimate_tokens closure"
    )
    return captured["estimate_tokens"]


async def routing_estimate(
    session,
    body_json: dict,
    model_config: dict,
    server_config: dict,
    session_id: str = "sess-parity",
) -> int:
    """Compute the routing estimate with the same tokenizer resolution the
    routing path uses in ``provider_with_fallback``."""
    tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
    import proxy.server as server_mod

    manager = MagicMock()
    manager.get = AsyncMock(return_value=session)
    with patch.object(server_mod, "session_manager", manager, create=True):
        estimate = await _estimate_effective_prompt_tokens_for_routing(
            make_request(session_id), body_json, tokenizer=tokenizer
        )
    if multiplier != 1.0:
        estimate = int(estimate * multiplier)
    return estimate


# ===================================================================
# AC2/AC3 — tokenizer parity: same message list -> same estimate
# ===================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_config",
    [NATIVE_MODEL_CONFIG, TIKTOKEN_MODEL_CONFIG],
    ids=["native-qwen3", "tiktoken-only"],
)
async def test_routing_and_compaction_estimates_agree(model_config):
    """Given the same session/config, routing and compaction estimates must
    be identical (LP-0MU5FSN3M006KKMO AC2/AC3)."""
    messages = make_messages(14)
    session = make_session(messages)
    body_json = {"model": "Qwen3", "messages": list(messages)}
    server_config = dict(SERVER_CONFIG)

    # Precondition: for the native tokenizer the fixture must actually
    # diverge from tiktoken, otherwise the parity assertion below is vacuous
    # (both paths would agree even with the buggy tiktoken-only wiring).
    if model_config is NATIVE_MODEL_CONFIG:
        native_tokenizer, _ = _get_tokenizer_for_model(model_config, server_config)
        assert _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=native_tokenizer
        ) != _estimate_prompt_tokens_for_routing({"messages": messages}), (
            "fixture must tokenize differently under native vs tiktoken for "
            "this parity test to be meaningful"
        )

    routing = await routing_estimate(session, body_json, model_config, server_config)
    compaction_fn = await capture_compaction_estimate(
        session, body_json, server_config, model_config
    )
    compaction = compaction_fn(messages)

    assert routing == compaction, (
        f"routing estimate ({routing}) != compaction estimate ({compaction}) "
        f"for model_config={model_config}"
    )


@pytest.mark.asyncio
async def test_native_tokenizer_used_by_both_paths():
    """With ``tokenizer: qwen3``, the compaction closure must use the native
    tokenizer, not tiktoken — i.e. it differs from the tiktoken-only count
    exactly as the routing estimate does."""
    messages = make_messages(12)
    session = make_session(messages)
    body_json = {"model": "Qwen3", "messages": list(messages)}

    native_fn = await capture_compaction_estimate(
        session, body_json, dict(SERVER_CONFIG), NATIVE_MODEL_CONFIG
    )
    tiktoken_fn = await capture_compaction_estimate(
        session, body_json, dict(SERVER_CONFIG), TIKTOKEN_MODEL_CONFIG
    )

    native_est = native_fn(messages)
    tiktoken_est = tiktoken_fn(messages)

    # The vendored Qwen3 tokenizer produces a higher count for prose than
    # cl100k; the two must not be accidentally identical (which would mean
    # the native tokenizer was not actually applied).
    assert native_est != tiktoken_est, (
        "native and tiktoken compaction estimates are identical — the native "
        "tokenizer is not being applied"
    )
    # Sanity: the native estimate matches a direct tiktoken-free count.
    direct_native = _estimate_prompt_tokens_for_routing(
        {"messages": messages}, tokenizer=_get_tokenizer_for_model(
            NATIVE_MODEL_CONFIG, SERVER_CONFIG
        )[0]
    )
    assert native_est == direct_native


# ===================================================================
# AC3 — input parity and superset relation
# ===================================================================


@pytest.mark.asyncio
async def test_routing_estimate_not_smaller_than_compaction():
    """Routing (``max(body, session)``) must never be smaller than the
    compaction estimate over the same produced history (AC3)."""
    messages = make_messages(14)
    session = make_session(messages)
    body_json = {"model": "Qwen3", "messages": list(messages)}

    routing = await routing_estimate(
        session, body_json, NATIVE_MODEL_CONFIG, dict(SERVER_CONFIG)
    )
    compaction_fn = await capture_compaction_estimate(
        session, body_json, dict(SERVER_CONFIG), NATIVE_MODEL_CONFIG
    )
    compaction = compaction_fn(messages)

    assert routing >= compaction, (
        f"routing estimate ({routing}) must be >= compaction estimate "
        f"({compaction}) — routing is the superset input"
    )


@pytest.mark.asyncio
async def test_delta_flow_compaction_uses_session_plus_delta():
    """Delta request: the compacted input is ``session.messages + delta``
    (the produced history), so compaction sees the full history even though
    the request body carries only the delta (AC3 input parity)."""
    from proxy.router_helpers import _handle_session

    stored = make_messages(6)
    delta = [
        {"role": "user", "content": "new turn question " * 60},
        {"role": "assistant", "content": "new turn answer " * 60},
    ]
    session = make_session(stored)
    body_json = {"model": "Qwen3", "messages": list(stored) + list(delta)}
    server_config = dict(SERVER_CONFIG)

    srv = make_server(session)
    srv.session_manager.compute_delta = MagicMock(return_value=(delta, True))
    captured: dict = {}

    def capture(srv_cap, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
        captured["messages"] = msgs
        captured["estimate_tokens"] = estimate_tokens
        return {
            "action": "noop",
            "applied": False,
            "dry_run": True,
            "messages": msgs,
            "reason": "dry_run",
        }

    with patch("proxy.lifecycle.get_model_config", return_value=NATIVE_MODEL_CONFIG), patch(
        "proxy.router_helpers._evaluate_session_compaction", side_effect=capture
    ):
        await _handle_session(srv, body_json, server_config, {})

    assert captured["messages"] == stored + delta, (
        "compaction must evaluate session.messages + delta (produced history)"
    )
    # The estimate over the produced history equals the estimate over the
    # full request body (the delta flow reconstructs it).
    assert captured["estimate_tokens"](captured["messages"]) == (
        captured["estimate_tokens"](stored + delta)
    )


# ===================================================================
# AC4 — observable agreement at the estimate boundary
# ===================================================================


@pytest.mark.asyncio
async def test_both_paths_cross_the_trigger_together():
    """A produced history above the trigger must be above it in BOTH paths;
    below it in both.  This is the estimate-level half of the observable
    agreement criterion (the decision-level half lives in the integration
    test LP-0MU5FTBOH005BR92)."""
    trigger = 58_300
    small = [{"role": "user", "content": "tiny"}]
    large = make_messages(400)  # far above the trigger with either tokenizer

    session_small = make_session(small)
    session_large = make_session(large)

    for messages, session, expected_over in (
        (small, session_small, False),
        (large, session_large, True),
    ):
        body_json = {"model": "Qwen3", "messages": list(messages)}
        routing = await routing_estimate(
            session, body_json, NATIVE_MODEL_CONFIG, dict(SERVER_CONFIG)
        )
        compaction_fn = await capture_compaction_estimate(
            session, body_json, dict(SERVER_CONFIG), NATIVE_MODEL_CONFIG
        )
        compaction = compaction_fn(messages)
        assert (routing > trigger) == expected_over
        assert (compaction > trigger) == expected_over
