"""
Live wiring tests for router_helpers._handle_session compaction path
(LP-0MTPMJGC3007YG0S — wire live Summarizer into dispatch).

Verifies that _handle_session passes a production Summarizer and a token
estimator through to _evaluate_session_compaction instead of the
always-None defaults that caused the Muse compaction hang (LP-0MTPK77WG009A4VH).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_server(**overrides):
    srv = MagicMock()
    cfg = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
            "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
            "summarizer_ctx_size": 8192,
            "summarizer_max_tokens": 512,
            **overrides,
        }
    }
    srv.config = cfg
    srv.logger = MagicMock()
    srv.session_manager = MagicMock()
    srv.session_manager.get_or_create = AsyncMock(
        return_value=(MagicMock(session_id="sess-wire", message_count=0), True)
    )
    return srv


def _make_session_messages(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYS"}]
    msgs.append({"role": "user", "content": "FIRST"})
    msgs.append({"role": "assistant", "content": "FR"})
    for i in range(num_turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _counting_estimator(messages):
    return 1000 * len(messages)


class TestHandleSessionCompactionWiring:
    @pytest.mark.asyncio
    async def test_compaction_wired_with_summarizer_and_estimator(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        body_json = {"model": "Qwen3", "messages": _make_session_messages(10)}
        server_config = srv.config["server"]
        captured = {}

        def capture(srv_cap, sid, msgs, mode, summarizer=None, estimate_tokens=None, **kw):
            captured["summarizer"] = summarizer
            captured["estimate_tokens"] = estimate_tokens
            captured["session_id"] = sid
            captured["mode"] = mode
            return {"action": "noop", "applied": False, "dry_run": True, "messages": msgs, "reason": "dry_run"}

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=capture):
            await _handle_session(srv, body_json, server_config, {})

        assert captured["summarizer"] is not None, "summarizer was None — compaction always noop/remote"
        assert callable(captured["summarizer"])
        assert captured["estimate_tokens"] is not None, "estimate_tokens was None"
        assert callable(captured["estimate_tokens"])

    @pytest.mark.asyncio
    async def test_compaction_receives_correct_session_id_and_messages(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        messages = _make_session_messages(5)
        body_json = {"model": "Qwen3", "messages": messages}
        server_config = srv.config["server"]
        captured = {}

        def capture(srv_cap, sid, msgs, mode, **kw):
            captured["session_id"] = sid
            captured["messages"] = msgs
            return {"action": "noop", "applied": False, "dry_run": True, "messages": msgs}

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=capture):
            await _handle_session(srv, body_json, server_config, {})

        assert captured["session_id"] == "sess-wire"
        assert captured["messages"] is not None

    @pytest.mark.asyncio
    async def test_compaction_mode_is_read_from_proxy_mode(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        body_json = {"model": "Qwen3", "messages": _make_session_messages(3)}
        server_config = srv.config["server"]
        captured = {}

        def capture(srv_cap, sid, msgs, mode, **kw):
            captured["mode"] = mode
            return {"action": "noop", "applied": False, "dry_run": True, "messages": msgs}

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=capture):
            await _handle_session(srv, body_json, server_config, {})

        assert captured["mode"] in ("fast", "cheap")

    @pytest.mark.asyncio
    async def test_summarizer_uses_server_config_from_srv(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server(llama_server_port=9999)
        body_json = {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}

        def fake_eval(*a, **kw):
            return {"action": "noop", "applied": False, "dry_run": True, "messages": kw.get("messages", a[2] if len(a) > 2 else [])}

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
            with patch("proxy.compaction_summarizer.build_local_summarizer") as mock_build:
                mock_build.return_value = MagicMock(return_value="fake_summary")
                await _handle_session(srv, body_json, srv.config["server"], {})
                assert mock_build.called

    @pytest.mark.asyncio
    async def test_summarizer_fallback_when_no_top_level_config(self):
        from proxy.router_helpers import _handle_session

        srv = MagicMock()
        srv.config = None
        srv.logger = MagicMock()
        srv.session_manager = MagicMock()
        srv.session_manager.get_or_create = AsyncMock(
            return_value=(MagicMock(session_id="s1", message_count=0), True)
        )
        body_json = {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}
        server_config = {"llama_server_port": 8080}

        def fake_eval(*a, **kw):
            return {"action": "noop", "applied": False, "dry_run": True, "messages": kw.get("messages", a[2] if len(a) > 2 else [])}

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=fake_eval):
            with patch("proxy.compaction_summarizer.build_local_summarizer") as mock_build:
                mock_build.return_value = MagicMock(return_value="")
                await _handle_session(srv, body_json, server_config, {})
                assert mock_build.called
                call_cfg = mock_build.call_args[0][0]
                assert call_cfg is not None

    @pytest.mark.asyncio
    async def test_compaction_exception_does_not_break_session(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        body_json = {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]}

        def raiser(srv_cap, sid, msgs, mode, **kw):
            raise RuntimeError("summarizer unreachable")

        with patch("proxy.router_helpers._evaluate_session_compaction", side_effect=raiser):
            result = await _handle_session(srv, body_json, srv.config["server"], {})

        assert result["session_id"] == "sess-wire"
        assert "compaction_applied" not in result

    def test_existing_dispatch_seam_still_works(self):
        from proxy.compaction import decide_session_compaction

        messages = _make_session_messages(60)
        config = {
            "server": {
                "local_model_ctx_size": 262144,
                "session_slot_pool_size": 3,
                "compaction_trigger_ratio": 0.70,
            }
        }
        decision = decide_session_compaction(
            messages, config, "fast", summarizer=None,
            estimate_tokens=_counting_estimator,
        )
        assert decision["action"] == "remote_with_guidance"
        assert decision["reason"] == "summarizer_unavailable"

        def small_summary(m):
            return "S" * 100

        decision = decide_session_compaction(
            messages, config, "fast",
            summarizer=small_summary,
            estimate_tokens=_counting_estimator,
        )
        assert decision["action"] == "compact"
