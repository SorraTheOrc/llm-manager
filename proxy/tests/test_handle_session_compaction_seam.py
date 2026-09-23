"""
Seam test: ``_handle_session`` delegates compaction to the shared helper.

Feature: LP-0MU5ARWSP001BYYB. Child: LP-0MU5IDK42009KNT5.

Locks in the extraction (LP-0MU5IBXD4000POV0): the former inline compaction
block inside ``_handle_session`` was replaced by a single call to
``evaluate_and_apply_compaction``. If someone re-inlines the block, this test
fails — keeping the bypass path and the local-dispatch path on the same
compaction implementation.

ACs covered:
  AC1 — inline block replaced by a call to the shared helper (exactly once
        per request).
  AC2 — the helper receives the same session context (result + session) so
        result propagation stays identical.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from proxy.router_helpers import _handle_session


def _make_server(**overrides):
    srv = MagicMock()
    srv.config = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
            **overrides,
        }
    }
    srv.logger = MagicMock()
    session = MagicMock(session_id="sess-seam", message_count=0)
    srv.session_manager = MagicMock()
    srv.session_manager.get_or_create = AsyncMock(
        return_value=(session, True)
    )
    srv.session_manager.update_messages = AsyncMock(return_value=None)
    srv.session_manager.mark_compacted = AsyncMock(return_value=None)
    return srv


def _noop_eval(srv, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
    return {
        "action": "noop", "applied": False, "dry_run": True,
        "messages": msgs, "reason": "below_trigger",
    }


class TestHandleSessionDelegatesToSharedHelper:
    @pytest.mark.asyncio
    async def test_handle_session_calls_shared_helper_exactly_once(self):
        """AC1: the inline block is replaced — the helper is invoked once."""
        srv = _make_server()
        body_json = {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}
        calls = []

        async def fake_helper(srv_arg, result, body_json_arg, server_config, **kw):
            calls.append(kw)
            return {
                "evaluated": True, "applied": False, "action": "noop",
                "reason": None, "estimated_before": 0, "estimated_after": 0,
            }

        with patch(
            "proxy.router_helpers.evaluate_and_apply_compaction",
            side_effect=fake_helper,
        ):
            await _handle_session(srv, body_json, srv.config["server"], {})

        assert len(calls) == 1, (
            "compaction must be evaluated exactly once per handled request"
        )

    @pytest.mark.asyncio
    async def test_shared_helper_receives_session_context(self):
        """AC2: the helper is handed the same ``result``/``session`` so the
        post-compaction propagation (body override, session update) can be
        identical to the former inline block."""
        srv = _make_server()
        body_json = {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}
        captured = {}

        async def fake_helper(srv_arg, result, body_json_arg, server_config, **kw):
            captured["result"] = result
            captured["session"] = kw.get("session")
            return {
                "evaluated": True, "applied": False, "action": "noop",
                "reason": None, "estimated_before": 0, "estimated_after": 0,
            }

        with patch(
            "proxy.router_helpers.evaluate_and_apply_compaction",
            side_effect=fake_helper,
        ):
            await _handle_session(srv, body_json, srv.config["server"], {})

        assert captured["result"]["session_id"] == "sess-seam"
        assert captured["session"].session_id == "sess-seam"

    @pytest.mark.asyncio
    async def test_live_compact_flows_through_helper_result_propagation(self):
        """AC2/AC3: a live compact decided by the helper propagates the
        exactly-once keys to ``_handle_session``'s result and body (no
        behavioural regression from removing the inline block)."""
        srv = _make_server()
        compacted = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "<summary>"},
        ]
        body_json = {
            "model": "Qwen3",
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
        }

        def compact_eval(srv_arg, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
            return {
                "action": "compact", "applied": True, "dry_run": False,
                "messages": compacted, "estimated_before": 90000,
                "estimated_after": 1000, "reason": "trigger_exceeded",
                "summary_text": "<summary>", "turns_summarized": 2,
                "recent_turns_kept": 1,
            }

        with patch(
            "proxy.router_helpers._evaluate_session_compaction",
            side_effect=compact_eval,
        ):
            result = await _handle_session(srv, body_json, srv.config["server"], {})

        assert result["compaction_applied"] is True
        assert result["compaction_estimated_before"] == 90000
        assert result["compaction_reason"] == "trigger_exceeded"
        assert result["compaction_summary_text"] == "<summary>"
        # The dispatched body after _handle_session is the compacted history.
        forwarded = result["body_override"]
        import json

        assert json.loads(forwarded)["messages"] == compacted
        # Session history updated to the compacted count.
        srv.session_manager.update_messages.assert_awaited_once()
        srv.session_manager.mark_compacted.assert_awaited_once()
