"""
Tests for the operator-overridable compaction summarizer system prompt
(LP-0MTTSL2AW000A5OG — R1 companion).

Verifies:
- The summarizer uses Pi's SUMMARIZATION_SYSTEM_PROMPT verbatim by default
  (role + guard rails only); the structured format template is appended to
  the serialised <conversation> in the USER message.
- An operator override file
  (``<repo>/.sorraAgents/prompts/compaction-summarizer.txt``) replaces the
  default system prompt without code changes (the proxy's custom
  system-prompt mechanism, mirroring ``proxy/proxy/prompt_resolver.py``).
- Fail-open: missing / oversized / invalid-UTF-8 / empty override files
  fall back to the default constant (warning logged, never crash).
- The committed repo template (``proxy/prompts/compaction-summarizer.txt``)
  stays equal to the code constant (single source of truth, no drift).
"""

import pathlib
from unittest.mock import MagicMock, patch

import pytest
from proxy.compaction_summarizer import (
    _MAX_SYSTEM_PROMPT_SIZE,
    _OVERRIDE_DIR,
    _SYSTEM_PROMPT_OVERRIDE_FILENAME,
    build_local_summarizer,
)
from proxy.provider import _SUMMARIZATION_PROMPT, _SUMMARIZER_SYSTEM_PROMPT


def _capture_body(config, messages):
    """Build the summarizer with httpx mocked and return the POST body."""
    with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value.__enter__.return_value = mock_client
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "S"}}]}
        mock_client.post.return_value = resp

        summarizer = build_local_summarizer(config or {}, llama_port=8080)
        assert summarizer(messages) == "S"
        args, kwargs = mock_client.post.call_args
        body = kwargs.get("json") or (args[1] if len(args) > 1 else kwargs.get("json"))
        return body


def _write_override(tmp_path, content_bytes, filename=_SYSTEM_PROMPT_OVERRIDE_FILENAME):
    path = tmp_path / filename
    if isinstance(content_bytes, str):
        path.write_text(content_bytes, encoding="utf-8")
    else:
        path.write_bytes(content_bytes)
    return path


class TestDefaultPromptShape:
    """Default (no override): Pi system prompt + template in user message."""

    def test_system_message_is_pi_system_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        body = _capture_body(None, [{"role": "user", "content": "hi"}])
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == _SUMMARIZER_SYSTEM_PROMPT

    def test_user_message_conversation_then_template(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        body = _capture_body(None, [{"role": "user", "content": "hi"}])
        msgs = body["messages"]
        user = next(m for m in msgs if m.get("role") == "user")["content"]
        assert user == f"<conversation>\nuser: hi\n</conversation>\n\n{_SUMMARIZATION_PROMPT}"

    def test_override_dir_follows_prompt_resolver_convention(self):
        """Default override dir is <repo>/.sorraAgents/prompts (no monkeypatch)."""
        assert _OVERRIDE_DIR.name == "prompts"
        assert _OVERRIDE_DIR.parent.name == ".sorraAgents"


class TestOperatorOverride:
    """<repo>/.sorraAgents/prompts/compaction-summarizer.txt replaces default."""

    def test_override_file_replaces_system_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        _write_override(tmp_path, "OPERATOR CUSTOM SYSTEM PROMPT")
        body = _capture_body(None, [{"role": "user", "content": "hi"}])
        assert body["messages"][0]["content"] == "OPERATOR CUSTOM SYSTEM PROMPT"

    def test_override_keeps_user_message_shape(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        _write_override(tmp_path, "CUSTOM")
        body = _capture_body(None, [{"role": "user", "content": "hi"}])
        user = next(m for m in body["messages"] if m.get("role") == "user")["content"]
        assert user == f"<conversation>\nuser: hi\n</conversation>\n\n{_SUMMARIZATION_PROMPT}"

    def test_oversized_override_falls_back(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        _write_override(tmp_path, "x" * (_MAX_SYSTEM_PROMPT_SIZE + 1))
        with caplog.at_level("WARNING", logger="proxy.compaction_summarizer"):
            body = _capture_body(None, [{"role": "user", "content": "hi"}])
        assert body["messages"][0]["content"] == _SUMMARIZER_SYSTEM_PROMPT
        assert any("exceeds" in r.message for r in caplog.records)

    def test_invalid_utf8_override_falls_back(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        _write_override(tmp_path, b"\xff\xfe\x00not-utf8")
        with caplog.at_level("WARNING", logger="proxy.compaction_summarizer"):
            body = _capture_body(None, [{"role": "user", "content": "hi"}])
        assert body["messages"][0]["content"] == _SUMMARIZER_SYSTEM_PROMPT
        assert any("UTF-8" in r.message for r in caplog.records)

    def test_empty_override_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.compaction_summarizer._OVERRIDE_DIR", tmp_path)
        _write_override(tmp_path, "   \n  ")
        body = _capture_body(None, [{"role": "user", "content": "hi"}])
        assert body["messages"][0]["content"] == _SUMMARIZER_SYSTEM_PROMPT


class TestRepoTemplateParity:
    """proxy/prompts/compaction-summarizer.txt must match the code constant."""

    def test_committed_template_matches_constant(self):
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        template_path = repo_root / "proxy" / "prompts" / "compaction-summarizer.txt"
        assert template_path.is_file(), "repo-default prompt file missing"
        content = template_path.read_text(encoding="utf-8").strip()
        assert content == _SUMMARIZER_SYSTEM_PROMPT
