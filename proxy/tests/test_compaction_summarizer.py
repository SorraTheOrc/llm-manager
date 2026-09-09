"""
Tests for proxy-side local Qwen3 summarizer (LP-0MTPMJG1P0038D32).

Verifies the production Summarizer callable backed by the local
llama-server: system-prompt wrapping, config-driven sizing,
fail-open on errors, and timeout handling.
"""
from unittest.mock import MagicMock, patch

import pytest
from proxy.provider import _SUMMARIZATION_PROMPT, _SUMMARIZER_SYSTEM_PROMPT


def _mock_success_response(content: str = "SUMMARY TEXT", status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _make_config(**overrides):
    cfg = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
            "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
            "summarizer_ctx_size": 8192,
            "summarizer_max_tokens": 512,
        }
    }
    cfg["server"].update(overrides)
    # Handle nested summarizer_model overrides
    if "summarizer_model" in overrides:
        cfg["server"]["summarizer_model"] = overrides["summarizer_model"]
    return cfg


class TestBuildLocalSummarizer:
    def test_returns_callable(self):
        from proxy.compaction_summarizer import build_local_summarizer

        s = build_local_summarizer({}, llama_port=8080)
        assert callable(s)

    def test_empty_middle_returns_empty_without_http(self):
        from proxy.compaction_summarizer import build_local_summarizer

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            s = build_local_summarizer({}, llama_port=8080)
            assert s([]) == ""
            mock_cls.assert_not_called()

    def test_calls_llama_server_with_correct_payload(self):
        from proxy.compaction_summarizer import build_local_summarizer

        cfg = _make_config()
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_success_response("hello summary")

            s = build_local_summarizer(cfg, llama_port=8080)
            result = s([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}])

            assert result == "hello summary"
            # Verify URL
            args, kwargs = mock_client.post.call_args
            assert "http://localhost:8080/v1/chat/completions" in args[0]
            body = kwargs.get("json") or args[1] if len(args) > 1 else kwargs.get("json")
            # Body checks
            assert body["model"] == "Qwen3"
            assert body["max_tokens"] == 512
            assert body["stream"] is False
            # System prompt present (Pi role + guard rails only)
            msgs = body["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == _SUMMARIZER_SYSTEM_PROMPT
            # User message: transcript serialised inside <conversation> tags,
            # then the structured format template appended (Pi's split)
            user_msg = next(m for m in msgs if m.get("role") == "user")
            user_content = user_msg["content"]
            assert user_content.startswith("<conversation>\n")
            assert "user: hello" in user_content
            assert "assistant: world" in user_content
            assert "\n</conversation>\n\n" in user_content
            assert _SUMMARIZATION_PROMPT in user_content

    def test_uses_config_values_for_model_and_max_tokens(self):
        from proxy.compaction_summarizer import build_local_summarizer

        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3-Next"},
                "summarizer_max_tokens": 256,
            }
        }
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_success_response("x")

            s = build_local_summarizer(cfg, llama_port=9090)
            s([{"role": "user", "content": "hi"}])
            body = mock_client.post.call_args[1].get("json") or mock_client.post.call_args[0][1]
            assert body["model"] == "Qwen3-Next"
            assert body["max_tokens"] == 256
            assert "http://localhost:9090/" in mock_client.post.call_args[0][0]

    def test_fail_open_on_connect_error(self):
        import httpx
        from proxy.compaction_summarizer import build_local_summarizer

        with (
            patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
            patch("proxy.compaction_summarizer.time.sleep"),
        ):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("refused")

            s = build_local_summarizer({}, llama_port=8080, timeout_seconds=2)
            # Must not raise, must return empty string after retries exhausted
            assert s([{"role": "user", "content": "hi"}]) == ""

    def test_fail_open_on_timeout(self):
        import httpx
        from proxy.compaction_summarizer import build_local_summarizer

        with (
            patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
            patch("proxy.compaction_summarizer.time.sleep"),
        ):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ReadTimeout("timeout")

            s = build_local_summarizer({}, llama_port=8080)
            assert s([{"role": "user", "content": "hi"}]) == ""

    def test_fail_open_on_non_200(self):
        from proxy.compaction_summarizer import build_local_summarizer

        with (
            patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
            patch("proxy.compaction_summarizer.time.sleep"),
        ):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "internal error"
            mock_client.post.return_value = resp

            s = build_local_summarizer({}, llama_port=8080)
            # 500 is transient — retried (default 2) then fail-open
            assert s([{"role": "user", "content": "hi"}]) == ""

    def test_fail_open_on_malformed_response(self):
        from proxy.compaction_summarizer import build_local_summarizer

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": []}  # no message
            mock_client.post.return_value = resp

            s = build_local_summarizer({}, llama_port=8080)
            assert s([{"role": "user", "content": "hi"}]) == ""

    def test_multimodal_content_handling(self):
        from proxy.compaction_summarizer import build_local_summarizer

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_success_response("ok")

            s = build_local_summarizer({}, llama_port=8080)
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello multimodal"},
                        {"type": "image_url", "image_url": {"url": "http://x"}},
                    ],
                }
            ]
            assert s(msgs) == "ok"
            body = mock_client.post.call_args[1].get("json") or mock_client.post.call_args[0][1]
            user_text = " ".join(m.get("content", "") for m in body["messages"] if m.get("role") == "user")
            assert "hello multimodal" in user_text

    def test_timeout_is_passed_to_client(self):
        from proxy.compaction_summarizer import build_local_summarizer

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_success_response("ok")

            s = build_local_summarizer({}, llama_port=8080, timeout_seconds=7)
            s([{"role": "user", "content": "hi"}])
            # Client constructed with timeout
            assert mock_cls.called
            _, kwargs = mock_cls.call_args
            timeout_arg = kwargs.get("timeout")
            assert timeout_arg is not None

    def test_integration_with_plan_session_compaction(self):
        """Summarizer integrates with the pure compaction planner."""
        from proxy.compaction import plan_session_compaction
        from proxy.compaction_summarizer import build_local_summarizer

        def fast_config():
            return {
                "server": {
                    "local_model_ctx_size": 262144,
                    "session_slot_pool_size": 3,
                    "compaction_trigger_ratio": 0.70,
                }
            }

        # Build a session large enough to trigger compaction
        msgs = [{"role": "system", "content": "SYS"}]
        msgs.append({"role": "user", "content": "FIRST"})
        msgs.append({"role": "assistant", "content": "FIRST_REPLY"})
        for i in range(60):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_success_response("INJECTED SUMMARY")

            summarizer = build_local_summarizer(fast_config(), llama_port=8080)

            # Use a generous estimator so trigger fires
            def big_estimator(m):
                return 1000 * len(m)

            result = plan_session_compaction(
                msgs, fast_config(), "fast",
                summarizer=summarizer,
                estimate_tokens=big_estimator,
            )
            assert result["action"] == "compact"
            assert result["summary_text"] == "INJECTED SUMMARY"
            # Summary marker present in compacted messages
            assert any(
                "The conversation history before this point was compacted" in str(m.get("content", ""))
                for m in result["messages"]
            )
