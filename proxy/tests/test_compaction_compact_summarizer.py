"""Tests for the remote-only ``compact`` compaction summarizer.

LP-0MTT0O74N009E7N2: ``build_compact_summarizer`` routes session-compaction
summarization through the ``models.compact`` chain (Muse via opencode-go →
DeepSeek via api.deepseek.com) instead of the local llama-server, so
summarization never contends with the GPU slots. These tests cover:

- provider config resolution and the local fallback (AC3),
- the Responses-API translation for the Muse tier (AC1/AC2),
- transparent Muse → DeepSeek fallback on HTTP error / empty response (AC2),
- fail-open (falsy ``""``) when every tier fails (AC2),
- per-tier timeout resolution (AC4),
- no ``localhost``/llama-server call on the remote path (AC2),
- the shipped config files declaring the chain with no local tier (AC1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from proxy import provider

LOGGER_NAME = "llama-proxy.compaction_summarizer"


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset provider cooldown/usage state so resolved chains are deterministic."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    yield


COMPACT_CONFIG: dict = {
    "models": {
        "compact": {
            "providers": [
                {
                    "name": "opencode-go-compact",
                    "type": "remote",
                    "provider": "opencode-go",
                    "endpoint": "https://opencode.ai/zen/go",
                    "api_key_env": "OPENCODE_API_KEY",
                    "model": "muse-spark-1.3-contributor",
                    "api": "openai-responses",
                    "forward_session_headers": False,
                },
                {
                    "name": "deepseek-flash-compact",
                    "type": "remote",
                    "provider": "deepseek",
                    "endpoint": "https://api.deepseek.com",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "model": "deepseek-flash",
                },
            ],
        }
    },
    "server": {
        "compaction_summarizer_timeout": 120,
        "summarizer_max_tokens": 512,
    },
}


def _chat_response(content: str | None, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = "body"
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _responses_response(content: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = "body"
    resp.json.return_value = {
        "id": "resp-1",
        "status": "completed",
        "model": "muse-spark-1.3-contributor",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
    }
    return resp


def _make_client(posts: list[dict], responses: list) -> MagicMock:
    """Build a patched ``httpx.Client`` recording posts and replaying responses."""
    client = MagicMock()

    def _post(url, json=None, headers=None, **kwargs):
        posts.append({"url": url, "json": json, "headers": headers})
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client.post.side_effect = _post
    cls = MagicMock()
    cls.return_value.__enter__.return_value = client
    cls.return_value.__exit__.return_value = False
    return cls


def _run(summarizer, messages, responses):
    posts: list[dict] = []
    cls = _make_client(posts, responses)
    with patch("proxy.compaction_summarizer.httpx.Client", cls):
        result = summarizer(messages)
    return result, posts, cls


_MESSAGES = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "world"},
]


class TestBuildCompactSummarizer:
    def test_returns_callable(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        assert callable(build_compact_summarizer(COMPACT_CONFIG))

    def test_local_fallback_when_no_compact_model(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        config = {"server": {"summarizer_model": {"type": "local", "llama_model": "Qwen3"}}}
        with patch("proxy.compaction_summarizer.build_local_summarizer") as mock_local:
            sentinel = MagicMock(return_value="local summary")
            mock_local.return_value = sentinel
            result = build_compact_summarizer(config, llama_port=9999)
            assert result is sentinel
            assert mock_local.called
            # The local fallback receives the caller's llama_port and config.
            assert mock_local.call_args.kwargs["llama_port"] == 9999

    def test_empty_messages_short_circuits_without_http(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            summarizer = build_compact_summarizer(COMPACT_CONFIG)
            assert summarizer([]) == ""
            mock_cls.assert_not_called()

    def test_muse_tier_uses_responses_api_and_synthesizes_session_header(self, monkeypatch):
        from proxy.compaction_summarizer import build_compact_summarizer

        monkeypatch.setenv("OPENCODE_API_KEY", "muse-key")
        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        result, posts, _ = _run(summarizer, _MESSAGES, [_responses_response("muse summary")])

        assert result == "muse summary"
        assert len(posts) == 1
        assert posts[0]["url"] == "https://opencode.ai/zen/go/v1/responses"
        body = posts[0]["json"]
        assert body["model"] == "muse-spark-1.3-contributor"
        assert body["max_output_tokens"] == 512
        assert "messages" not in body and "input" in body
        assert posts[0]["headers"]["Authorization"] == "Bearer muse-key"
        assert posts[0]["headers"]["x-opencode-session"].startswith("compact-")

    def test_falls_back_to_deepseek_on_muse_http_error(self, monkeypatch, caplog):
        from proxy.compaction_summarizer import build_compact_summarizer

        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            result, posts, _ = _run(
                summarizer,
                _MESSAGES,
                [_chat_response(None, status=500), _chat_response("deepseek summary")],
            )

        assert result == "deepseek summary"
        assert [p["url"] for p in posts] == [
            "https://opencode.ai/zen/go/v1/responses",
            "https://api.deepseek.com/v1/chat/completions",
        ]
        assert posts[1]["json"]["model"] == "deepseek-flash"
        assert posts[1]["headers"]["Authorization"] == "Bearer ds-key"
        assert any(
            "opencode-go-compact" in rec.message and "http_500" in rec.message
            for rec in caplog.records
        )

    def test_falls_back_to_deepseek_on_muse_empty_response(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        result, posts, _ = _run(
            summarizer,
            _MESSAGES,
            [_responses_response(""), _chat_response("deepseek summary")],
        )
        assert result == "deepseek summary"
        assert len(posts) == 2

    def test_falls_back_to_deepseek_on_muse_timeout(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        result, posts, _ = _run(
            summarizer,
            _MESSAGES,
            [httpx.ReadTimeout("slow"), _chat_response("deepseek summary")],
        )
        assert result == "deepseek summary"
        assert len(posts) == 2

    def test_fail_open_when_all_tiers_fail(self, caplog):
        from proxy.compaction import EmptySummary
        from proxy.compaction_summarizer import build_compact_summarizer

        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            result, posts, _ = _run(
                summarizer,
                _MESSAGES,
                [_chat_response(None, status=500), _chat_response(None, status=503)],
            )

        assert result == ""
        assert isinstance(result, EmptySummary)
        assert result.kind == "http_503"
        assert result.attempts == 2
        assert len(posts) == 2

    def test_per_tier_timeout_respects_configured_value(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        _, _, cls = _run(summarizer, _MESSAGES, [_responses_response("summary")])
        # httpx.Timeout(120) is passed as the client timeout.
        timeout = cls.call_args.kwargs["timeout"]
        assert timeout.connect == 120.0
        assert timeout.read == 120.0

    def test_per_tier_timeout_override_wins(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        config = {
            "models": {
                "compact": {
                    "timeout_seconds": 45,
                    "providers": COMPACT_CONFIG["models"]["compact"]["providers"],
                }
            },
            "server": {"compaction_summarizer_timeout": 120},
        }
        summarizer = build_compact_summarizer(config)
        _, _, cls = _run(summarizer, _MESSAGES, [_responses_response("summary")])
        timeout = cls.call_args.kwargs["timeout"]
        assert timeout.connect == 45.0

    def test_never_calls_local_llama_server(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        _, posts, _ = _run(summarizer, _MESSAGES, [_responses_response("summary")])
        assert all("localhost" not in p["url"] for p in posts)
        assert all(":8080" not in p["url"] for p in posts)

    def test_file_operations_appended_to_remote_summary(self):
        from proxy.compaction_summarizer import build_compact_summarizer

        messages = [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "/tmp/a.py"}',
                        }
                    }
                ],
            },
        ]
        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        result, _, _ = _run(summarizer, messages, [_responses_response("summary")])
        assert result.startswith("summary")
        assert "<read-files>\n/tmp/a.py\n</read-files>" in result

    def test_no_provider_returns_empty_summary(self):
        from proxy.compaction import EmptySummary
        from proxy.compaction_summarizer import build_compact_summarizer

        # Both tiers are in cooldown -> resolve_provider returns None.
        provider._provider_unavailable_until["opencode-go-compact"] = 10**12
        provider._provider_unavailable_until["deepseek-flash-compact"] = 10**12
        summarizer = build_compact_summarizer(COMPACT_CONFIG)
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            result = summarizer(_MESSAGES)
            mock_cls.assert_not_called()
        assert isinstance(result, EmptySummary)
        assert result.kind == "no_provider"


class TestShippedCompactConfig:
    @pytest.mark.parametrize(
        "config_name",
        ["config.yaml", "config-fast.yaml", "config-cheap.yaml"],
    )
    def test_compact_model_declares_remote_chain(self, config_name):
        import yaml

        config_path = Path(__file__).resolve().parents[2] / "proxy" / config_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        compact = config["models"]["compact"]
        providers = compact["providers"]

        assert [p["name"] for p in providers] == [
            "opencode-go-compact",
            "deepseek-flash-compact",
        ]
        muse, deepseek = providers
        assert muse["type"] == "remote" and muse["provider"] == "opencode-go"
        assert muse["endpoint"] == "https://opencode.ai/zen/go"
        assert muse["model"] == "muse-spark-1.3-contributor"
        assert muse["api"] == "openai-responses"
        assert muse["forward_session_headers"] is False
        assert deepseek["type"] == "remote" and deepseek["provider"] == "deepseek"
        assert deepseek["endpoint"] == "https://api.deepseek.com"
        assert deepseek["api_key_env"] == "DEEPSEEK_API_KEY"
        # Strictly remote: no local tier may contend with the GPU slots.
        assert all(p["type"] != "local" for p in providers)
        assert "llama_model" not in muse and "llama_model" not in deepseek
