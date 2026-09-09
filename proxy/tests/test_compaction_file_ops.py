"""
Tests for file-operation tracking in proxy compaction summaries (R2,
LP-0MTTPXI1Y003YFOU).

Verifies:
- File operations (read / write / edit tool calls) are extracted from the
  assistant tool_calls in the summarizer's middle messages.
- Extracted paths are deduplicated; files both read and modified are listed
  only under <modified-files> (Pi semantics).
- Summaries embed sorted ``<read-files>`` / ``<modified-files>`` XML sections
  when operations exist and omit them entirely when none exist.
- Extraction is defensive: malformed arguments, missing paths, non-file
  tools, and non-assistant messages never crash or pollute the lists.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from proxy.compaction_summarizer import (
    build_local_summarizer,
    extract_file_operations,
    format_file_operations,
)


def _tool_call(name: str, path: str | None = None, raw_args: str | None = None):
    """Build an OpenAI-style assistant tool_call dict.

    ``path`` is serialised into JSON arguments; pass ``raw_args`` to simulate
    malformed/non-JSON or non-path payloads.
    """
    if raw_args is None:
        raw_args = json.dumps({"path": path}) if path is not None else "{}"
    return {
        "id": f"call_{name}_{path or 'x'}",
        "type": "function",
        "function": {"name": name, "arguments": raw_args},
    }


def _assistant(*tool_calls):
    return {"role": "assistant", "content": "work", "tool_calls": list(tool_calls)}


class TestExtractFileOperations:
    def test_read_tool_goes_to_read_set(self):
        ops = extract_file_operations([_assistant(_tool_call("read", "a.ts"))])
        assert ops["read"] == {"a.ts"}
        assert ops["modified"] == set()

    def test_write_and_edit_go_to_modified_set(self):
        msgs = [
            _assistant(_tool_call("write", "b.txt"), _tool_call("edit", "c.py")),
        ]
        ops = extract_file_operations(msgs)
        assert ops["modified"] == {"b.txt", "c.py"}
        assert ops["read"] == set()

    def test_read_then_modified_counts_as_modified_only(self):
        """A file read AND modified is reported under modified-files only."""
        msgs = [
            _assistant(_tool_call("read", "a.ts"), _tool_call("edit", "a.ts")),
        ]
        ops = extract_file_operations(msgs)
        assert ops["modified"] == {"a.ts"}
        assert ops["read"] == set()

    def test_duplicate_reads_are_deduplicated(self):
        msgs = [_assistant(_tool_call("read", "a.ts"), _tool_call("read", "a.ts"))]
        ops = extract_file_operations(msgs)
        assert ops["read"] == {"a.ts"}

    def test_non_file_tools_ignored(self):
        msgs = [
            _assistant(
                _tool_call("read", "a.ts"),
                _tool_call("bash", None, raw_args=json.dumps({"command": "ls"})),
                _tool_call("grep", None, raw_args=json.dumps({"pattern": "x"})),
            )
        ]
        ops = extract_file_operations(msgs)
        assert ops["read"] == {"a.ts"}
        assert ops["modified"] == set()

    def test_malformed_json_arguments_skipped(self):
        msgs = [_assistant(_tool_call("read", None, raw_args="{not json"))]
        ops = extract_file_operations(msgs)
        assert ops["read"] == set()
        assert ops["modified"] == set()

    def test_missing_path_skipped(self):
        msgs = [_assistant(_tool_call("edit", None, raw_args=json.dumps({"old": "x"})))]
        ops = extract_file_operations(msgs)
        assert ops["modified"] == set()

    def test_arguments_as_dict_supported(self):
        """Some clients pass arguments already parsed as a dict."""
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": {"path": "d.ts"}},
        }
        ops = extract_file_operations([_assistant(tc)])
        assert ops["read"] == {"d.ts"}

    def test_non_assistant_and_tool_result_messages_ignored(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_x", "content": "output"},
            _assistant(_tool_call("read", "a.ts")),
        ]
        ops = extract_file_operations(msgs)
        assert ops["read"] == {"a.ts"}

    def test_empty_messages(self):
        ops = extract_file_operations([])
        assert ops["read"] == set()
        assert ops["modified"] == set()


class TestFormatFileOperations:
    def test_both_sections_sorted(self):
        out = format_file_operations({"z.ts", "a.ts"}, {"m.py", "b.txt"})
        assert out == (
            "\n\n<read-files>\na.ts\nz.ts\n</read-files>\n\n<modified-files>\nb.txt\nm.py\n</modified-files>"
        )

    def test_read_only(self):
        out = format_file_operations({"a.ts"}, set())
        assert out == "\n\n<read-files>\na.ts\n</read-files>"

    def test_modified_only(self):
        out = format_file_operations(set(), {"b.txt"})
        assert out == "\n\n<modified-files>\nb.txt\n</modified-files>"

    def test_empty_returns_empty_string(self):
        assert format_file_operations(set(), set()) == ""


class TestSummarizerEmbedsFileOps:
    """build_local_summarizer appends file-op XML to the model's summary."""

    def _summarize(self, middle_messages):
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "SUMMARY BODY"}}]}
            mock_client.post.return_value = resp
            summarizer = build_local_summarizer({}, llama_port=8080)
            return summarizer(middle_messages)

    def test_file_ops_appended_when_present(self):
        msgs = [
            _assistant(_tool_call("read", "b.ts"), _tool_call("edit", "a.ts")),
            {"role": "user", "content": "keep going"},
        ]
        out = self._summarize(msgs)
        assert out.startswith("SUMMARY BODY")
        assert out.endswith("<read-files>\nb.ts\n</read-files>\n\n<modified-files>\na.ts\n</modified-files>")

    def test_no_xml_when_no_file_ops(self):
        out = self._summarize([{"role": "user", "content": "plain turn"}])
        assert out == "SUMMARY BODY"
        assert "<read-files>" not in out
        assert "<modified-files>" not in out

    def test_empty_model_content_returns_empty_no_xml(self):
        """Fail-open on empty model content: no partial file-op-only output."""
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
            mock_client.post.return_value = resp
            summarizer = build_local_summarizer({}, llama_port=8080)
            out = summarizer([_assistant(_tool_call("read", "a.ts"))])
            assert out == ""


class TestFileOpsFlowThroughPlanner:
    """plan_session_compaction summary_text carries the file-op XML."""

    def test_summary_text_includes_file_ops(self):
        from proxy.compaction import plan_session_compaction

        msgs = [{"role": "system", "content": "SYS"}]
        msgs.append({"role": "user", "content": "FIRST"})
        for i in range(60):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"a{i}",
                    "tool_calls": [_tool_call("read", "file.ts")] if i == 2 else [],
                }
            )

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "CHECKPOINT"}}]}
            mock_client.post.return_value = resp
            summarizer = build_local_summarizer({}, llama_port=8080)

            def big_estimator(m):
                return 1000 * len(m)

            result = plan_session_compaction(
                msgs,
                {
                    "server": {
                        "local_model_ctx_size": 262144,
                        "session_slot_pool_size": 3,
                        "compaction_trigger_ratio": 0.70,
                    }
                },
                "fast",
                summarizer=summarizer,
                estimate_tokens=big_estimator,
            )
            assert result["action"] == "compact"
            assert "CHECKPOINT" in result["summary_text"]
            assert "<read-files>\nfile.ts\n</read-files>" in result["summary_text"]
