"""
Unit tests for the shared compaction helper ``evaluate_and_apply_compaction``.

Feature: LP-0MU5ARWSP001BYYB (compaction unreachable for oversized sessions).
Child: LP-0MU5IBXD4000POV0 (extract shared compaction evaluation helper).

The helper is the single seam both the local-dispatch path (``_handle_session``)
and the router bypass path (``_proxy_with_fallback_cycle``) call. These tests
pin the seam's contract directly:

  AC1 — helper exists and encapsulates the decision/apply logic.
  AC2 — identical result-dict structure to the former inline block.
  AC3 — idempotent gating via ``result["compaction_evaluated"]``.
  AC4 — summarizer/estimator are injectable (production callables built when
        omitted).
  AC5 — fail-open error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from proxy.router_helpers import evaluate_and_apply_compaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(overrides: dict | None = None):
    srv = MagicMock()
    srv.config = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
            **(overrides or {}),
        }
    }
    srv.logger = MagicMock()
    srv.session_manager = MagicMock()
    srv.session_manager.update_messages = AsyncMock(return_value=None)
    srv.session_manager.mark_compacted = AsyncMock(return_value=None)
    return srv


def _counting_estimator(messages):
    return 1000 * len(messages)


def _messages(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(num_turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


# ---------------------------------------------------------------------------
# AC1 / AC2 — helper contract and result structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_exists_and_returns_outcome_contract():
    """AC1/AC2: the helper is importable and returns the documented outcome."""
    srv = _make_server()
    result = {"session_id": "sess-1"}
    body_json = {"messages": _messages(2)}

    def fake_eval(*_a, **_kw):
        return {
            "action": "noop",
            "applied": False,
            "dry_run": True,
            "messages": body_json["messages"],
            "reason": "below_trigger",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        outcome = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    assert outcome["evaluated"] is True
    assert outcome["action"] == "noop"
    assert outcome["applied"] is False
    assert outcome["reason"] == "below_trigger"
    assert set(outcome) >= {
        "evaluated", "action", "applied", "dry_run",
        "estimated_before", "estimated_after", "reason", "messages",
    }


@pytest.mark.asyncio
async def test_live_compact_mutates_result_and_body_json():
    """AC2: a live compact sets exactly the former inline-block keys and
    rewrites body_json messages to the compacted history."""
    srv = _make_server()
    result = {"session_id": "sess-live"}
    original = _messages(40)
    compacted = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "<summary>"}]
    body_json = {"model": "Qwen3", "messages": original}

    def fake_eval(*_a, **_kw):
        return {
            "action": "compact",
            "applied": True,
            "dry_run": False,
            "messages": compacted,
            "estimated_before": 80000,
            "estimated_after": 2000,
            "reason": "trigger_exceeded",
            "mode": "fast",
            "summary_text": "<summary>",
            "turns_summarized": 38,
            "recent_turns_kept": 2,
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        outcome = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    # Same keys the inline block produced.
    assert result["compaction_applied"] is True
    assert result["compaction_estimated_before"] == 80000
    assert result["compaction_reason"] == "trigger_exceeded"
    assert result["compaction_summary_text"] == "<summary>"
    assert result["compaction_turns_summarized"] == 38
    assert result["compaction_recent_turns_kept"] == 2
    assert result["body_override"] is not None
    # Body rewritten to the compacted history.
    assert body_json["messages"] == compacted
    assert outcome["applied"] is True
    assert outcome["messages"] == compacted


@pytest.mark.asyncio
async def test_remote_with_guidance_sets_flag():
    """AC2: live ``remote_with_guidance`` sets the escalation flag."""
    srv = _make_server()
    result = {"session_id": "sess-guidance"}
    body_json = {"messages": _messages(40)}

    def fake_eval(*_a, **_kw):
        return {
            "action": "remote_with_guidance",
            "applied": False,
            "dry_run": False,
            "messages": body_json["messages"],
            "estimated_before": 120000,
            "reason": "summarizer_unavailable",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        outcome = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    assert result["compaction_remote_with_guidance"] is True
    assert result["compaction_reason"] == "summarizer_unavailable"
    assert outcome["action"] == "remote_with_guidance"
    assert outcome["applied"] is False


# ---------------------------------------------------------------------------
# AC3 — idempotent gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_call_is_a_noop():
    """AC3: a second call in the same request cycle short-circuits."""
    srv = _make_server()
    result = {"session_id": "sess-idem"}
    body_json = {"messages": _messages(40)}
    calls = []

    def fake_eval(*_a, **_kw):
        calls.append(1)
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": body_json["messages"], "reason": "below_trigger",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        first = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )
        second = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    assert first["evaluated"] is True
    assert second["evaluated"] is False
    assert second["reason"] == "already_evaluated"
    assert len(calls) == 1, "compaction must only evaluate once per cycle"


@pytest.mark.asyncio
async def test_preexisting_flag_short_circuits():
    """AC3: a pre-set ``compaction_evaluated`` flag prevents re-evaluation
    (the mechanism the bypass path uses to signal local dispatch)."""
    srv = _make_server()
    result = {"session_id": "sess-flag", "compaction_evaluated": True}
    body_json = {"messages": _messages(40)}
    calls = []

    with patch(
        "proxy.router_helpers._evaluate_session_compaction",
        side_effect=lambda *a, **k: calls.append(1),
    ):
        outcome = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    assert outcome["evaluated"] is False
    assert outcome["reason"] == "already_evaluated"
    assert calls == []


# ---------------------------------------------------------------------------
# AC4 — injectable summarizer / estimator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injected_summarizer_and_estimator_are_used():
    """AC4: injected callables are passed through to ``_evaluate_session_compaction``."""
    srv = _make_server()
    result = {"session_id": "sess-inject"}
    body_json = {"model": "Qwen3", "messages": _messages(3)}
    captured = {}

    def summarizer(middle, previous_summary=None):
        return "SUM"

    def fake_eval(_srv, _sid, _msgs, _mode, summarizer=None, estimate_tokens=None, **kw):
        captured["summarizer"] = summarizer
        captured["estimate_tokens"] = estimate_tokens
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": _msgs, "reason": "below_trigger",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            summarizer=summarizer,
            estimate_tokens=_counting_estimator,
        )

    assert captured["summarizer"] is summarizer
    assert captured["estimate_tokens"] is _counting_estimator


@pytest.mark.asyncio
async def test_production_estimator_built_when_omitted():
    """AC4: when ``estimate_tokens`` is omitted, the helper builds the real
    tokenizer-based estimator and passes it through."""
    srv = _make_server()
    result = {"session_id": "sess-prod"}
    body_json = {"model": "Qwen3", "messages": _messages(3)}
    captured = {}

    def fake_eval(_srv, _sid, _msgs, _mode, summarizer=None, estimate_tokens=None, **kw):
        captured["estimate_tokens"] = estimate_tokens
        captured["summarizer"] = summarizer
        return {
            "action": "noop", "applied": False, "dry_run": True,
            "messages": _msgs, "reason": "below_trigger",
        }

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
        await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
        )

    assert callable(captured["estimate_tokens"])
    assert captured["summarizer"] is not None


# ---------------------------------------------------------------------------
# AC5 — fail-open error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluation_exception_fails_open_and_marks_evaluated():
    """AC5: an exception is swallowed, dispatch is unchanged, and the
    evaluation is still marked so it is not retried this cycle."""
    srv = _make_server()
    result = {"session_id": "sess-boom"}
    original = _messages(5)
    body_json = {"messages": list(original)}

    def boom(*_a, **_kw):
        raise RuntimeError("summarizer exploded")

    with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=boom):
        outcome = await evaluate_and_apply_compaction(
            srv, result, body_json, srv.config["server"],
            estimate_tokens=_counting_estimator,
        )

    assert outcome["evaluated"] is True
    assert outcome["reason"] == "evaluation_failed"
    # Dispatch body untouched.
    assert body_json["messages"] == original
    assert result["compaction_evaluated"] is True
    srv.logger.warning.assert_called()
