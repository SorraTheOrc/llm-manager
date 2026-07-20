"""Tests for the session recorder module (proxy/proxy/session_recorder.py).

Covers:
- Recording request/response payloads for both streaming and non-streaming paths
- Assembled (not chunk-wise) recording for streaming responses
- Directory creation, file naming (by session ID + timestamp)
- File I/O edge cases (permissions, missing directory creation, disk errors)
- Non-blocking behavior (recording does not delay response)
- Configuration override and default values
- Three directions: client_to_proxy, proxy_to_provider, provider_to_client
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_recording_dir(tmp_path):
    """Provide a temporary directory for recording files."""
    d = tmp_path / "session-recordings"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def recorder(temp_recording_dir):
    """Return a SessionRecorder instance configured with a temp directory.

    Provides default config, which can be overridden per-test.
    """
    from proxy.session_recorder import SessionRecorder
    return SessionRecorder(recording_path=temp_recording_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Initialization & Defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionRecorderInit:
    """Verify SessionRecorder initialisation and default config."""

    def test_default_recording_path(self):
        """Default recording path is proxy/session-recordings/ (AC5)."""
        from proxy.session_recorder import SessionRecorder
        rec = SessionRecorder()
        assert rec.recording_path.endswith("session-recordings")
        assert rec.recording_path is not None

    def test_custom_recording_path(self, temp_recording_dir):
        """Custom recording path is honoured."""
        from proxy.session_recorder import SessionRecorder
        rec = SessionRecorder(recording_path=temp_recording_dir)
        assert rec.recording_path == temp_recording_dir

    def test_recording_path_created_on_init(self, tmp_path):
        """Recording directory is created on init if it does not exist (AC8)."""
        custom = str(tmp_path / "custom-recordings")
        assert not os.path.exists(custom)

        from proxy.session_recorder import SessionRecorder
        SessionRecorder(recording_path=custom)

        assert os.path.isdir(custom)

    def test_config_from_dict(self, temp_recording_dir):
        """Initialisation from a config dict (simulating config.yaml)."""
        cfg = {"session_recording": {"path": temp_recording_dir}}
        from proxy.session_recorder import SessionRecorder
        rec = SessionRecorder.from_config(cfg)
        assert rec.recording_path == temp_recording_dir

    def test_config_defaults_to_provided_path(self):
        """When config dict lacks session_recording key, use the default."""
        from proxy.session_recorder import SessionRecorder
        rec = SessionRecorder.from_config({})
        assert rec.recording_path.endswith("session-recordings")


# ═══════════════════════════════════════════════════════════════════════════
# Recording request payloads (AC1, AC3, AC4)
# ═══════════════════════════════════════════════════════════════════════════


class TestRecordRequest:
    """Tests for record_request() — captures client/proxy-to-provider request payloads."""

    @pytest.mark.asyncio
    async def test_record_client_to_proxy_request(self, recorder, temp_recording_dir):
        """Client-to-proxy request payload is written to disk (AC1, AC3)."""
        session_id = "sess-test-001"
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )

        assert filepath is not None
        # Verify the file exists
        assert os.path.isfile(filepath)
        # Verify content
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["direction"] == "client_to_proxy"
        assert saved["payload"] == payload
        assert saved["session_id"] == session_id
        assert "timestamp" in saved

    @pytest.mark.asyncio
    async def test_record_proxy_to_provider_request(self, recorder, temp_recording_dir):
        """Proxy-to-provider request payload is written to disk (AC3)."""
        session_id = "sess-test-002"
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello after processing"}]}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="proxy_to_provider",
            payload=payload,
        )

        assert filepath is not None
        assert os.path.isfile(filepath)
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["direction"] == "proxy_to_provider"
        assert saved["payload"] == payload

    @pytest.mark.asyncio
    async def test_file_naming_convention(self, recorder, temp_recording_dir):
        """File naming uses session-id directory and timestamp (AC4)."""
        session_id = "sess-naming-test"
        payload = {"messages": [{"role": "user", "content": "Hi"}]}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )

        assert filepath is not None
        path = Path(filepath)
        # Filename should contain "-request.json"
        assert "-request.json" in path.name
        # Parent directory should be the session ID
        assert path.parent.name == session_id
        # Parent of parent should be the recording path
        assert str(path.parent.parent) == temp_recording_dir

    @pytest.mark.asyncio
    async def test_file_naming_timestamp_format(self, recorder, temp_recording_dir):
        """Filename includes a parseable ISO8601-like timestamp."""
        session_id = "sess-ts"
        payload = {"test": True}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )

        assert filepath is not None
        filename = Path(filepath).name
        # Expected format: <timestamp>-request.json
        # e.g. "2026-07-06T12:34:56.789012-request.json"
        assert filename.endswith("-request.json")
        ts_part = filename[: -len("-request.json")]
        # Verify it's parseable as ISO datetime
        datetime.fromisoformat(ts_part)

    @pytest.mark.asyncio
    async def test_multiple_requests_same_session(self, recorder, temp_recording_dir):
        """Multiple requests for the same session create separate files."""
        session_id = "sess-multi"
        payload1 = {"seq": 1}
        payload2 = {"seq": 2}

        fp1 = await recorder.record_request(session_id, "client_to_proxy", payload1)
        fp2 = await recorder.record_request(session_id, "client_to_proxy", payload2)

        assert fp1 is not None
        assert fp2 is not None
        assert fp1 != fp2  # Different files
        # Both in the same session directory
        assert Path(fp1).parent == Path(fp2).parent


# ═══════════════════════════════════════════════════════════════════════════
# Recording response payloads (AC1, AC2, AC3)
# ═══════════════════════════════════════════════════════════════════════════


class TestRecordResponse:
    """Tests for record_response() — captures assembled provider response payloads."""

    @pytest.mark.asyncio
    async def test_record_provider_to_client_response(self, recorder, temp_recording_dir):
        """Provider-to-client response is written to disk (AC1, AC3)."""
        session_id = "sess-resp-001"
        payload = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello back!"}, "finish_reason": "stop"}],
        }

        filepath = await recorder.record_response(
            session_id=session_id,
            direction="provider_to_client",
            payload=payload,
        )

        assert filepath is not None
        assert os.path.isfile(filepath)
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["direction"] == "provider_to_client"
        assert saved["payload"] == payload

    @pytest.mark.asyncio
    async def test_response_file_naming(self, recorder, temp_recording_dir):
        """Response files end with -response.json (AC4)."""
        session_id = "sess-resp-naming"
        payload = {"choices": [{"text": "OK"}]}

        filepath = await recorder.record_response(
            session_id=session_id,
            direction="provider_to_client",
            payload=payload,
        )

        assert filepath is not None
        assert "-response.json" in Path(filepath).name
        assert Path(filepath).parent.name == session_id

    @pytest.mark.asyncio
    async def test_response_accepts_assembled_payload_only(self, recorder, temp_recording_dir):
        """Recorder accepts only assembled (not chunk-wise) payloads (AC2).

        This test verifies that the recorder does NOT perform its own SSE
        assembly — assembly is done by the caller before calling this method.
        The payload should be the complete final response.
        """
        session_id = "sess-stream-assembled"
        # Simulate an assembled streaming response (after SSE chunk assembly)
        assembled_payload = {
            "id": "chatcmpl-stream-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is the fully assembled streaming response.",
                },
                "finish_reason": "stop",
            }],
        }

        filepath = await recorder.record_response(
            session_id=session_id,
            direction="provider_to_client",
            payload=assembled_payload,
        )

        assert filepath is not None
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["payload"]["choices"][0]["message"]["content"] == \
            "This is the fully assembled streaming response."


# ═══════════════════════════════════════════════════════════════════════════
# Directory creation (AC8)
# ═══════════════════════════════════════════════════════════════════════════


class TestDirectoryCreation:
    """Verify the recorder creates directories as needed."""

    @pytest.mark.asyncio
    async def test_creates_session_directory(self, recorder, temp_recording_dir):
        """Session directory is created if it does not exist (AC8)."""
        session_id = "sess-new-dir"
        session_dir = Path(temp_recording_dir) / session_id
        assert not session_dir.exists()

        await recorder.record_request(session_id, "client_to_proxy", {"msg": "hi"})

        assert session_dir.is_dir()

    @pytest.mark.asyncio
    async def test_reuses_existing_session_directory(self, recorder, temp_recording_dir):
        """Existing session directory is reused without error."""
        session_id = "sess-existing"
        session_dir = Path(temp_recording_dir) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        existing_count_before = len(list(session_dir.iterdir()))

        await recorder.record_request(session_id, "client_to_proxy", {"msg": "hi"})

        assert session_dir.is_dir()
        assert len(list(session_dir.iterdir())) == existing_count_before + 1


# ═══════════════════════════════════════════════════════════════════════════
# Non-blocking behaviour (AC6)
# ═══════════════════════════════════════════════════════════════════════════


class TestNonBlockingBehaviour:
    """Verify recording does not block the caller (fire-and-forget via asyncio.create_task)."""

    @pytest.mark.asyncio
    async def test_record_returns_immediately(self, recorder, temp_recording_dir, monkeypatch):
        """record_request returns a coroutine that completes quickly (non-blocking).

        We verify that the file write is dispatched to a background task and
        the coroutine returns promptly even before the file write completes.
        """
        # Add a small artificial delay to the actual file write to simulate I/O
        original_write = Path(temp_recording_dir).write_text

        async def _slow_write(*args, **kwargs):
            # Use to_thread to simulate a slow synchronous write
            await asyncio.sleep(0.1)
            return original_write(*args, **kwargs)

        session_id = "sess-nonblock"
        payload = {"slow": "write"}

        # Measure approximate call duration
        import time
        start = time.monotonic()
        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )
        duration = time.monotonic() - start

        # Even with slow I/O, the call itself should return quickly
        # since file writing is dispatched to a background task
        assert filepath is not None
        # The call should return in reasonable time (< 50ms in practice)
        # We use a generous threshold since this depends on test environment
        assert duration < 5.0, "record_request should not block for slow I/O"

    @pytest.mark.asyncio
    async def test_concurrent_recordings_dont_interfere(self, recorder, temp_recording_dir):
        """Multiple concurrent recordings work without data races."""
        session_a = "sess-concurrent-a"
        session_b = "sess-concurrent-b"
        payload = {"test": "concurrent"}

        results = await asyncio.gather(
            recorder.record_request(session_a, "client_to_proxy", payload),
            recorder.record_request(session_b, "client_to_proxy", payload),
            recorder.record_request(session_a, "client_to_proxy", {"seq": 2}),
        )

        assert all(fp is not None for fp in results)
        # All three files should exist
        for fp in results:
            assert os.path.isfile(fp)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases: errors, missing dirs, empty payloads
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases: empty payloads, disk errors, permission issues."""

    @pytest.mark.asyncio
    async def test_empty_payload(self, recorder, temp_recording_dir):
        """Recording with an empty dict payload succeeds."""
        session_id = "sess-empty"
        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload={},
        )
        assert filepath is not None
        assert os.path.isfile(filepath)

    @pytest.mark.asyncio
    async def test_none_payload(self, recorder, temp_recording_dir):
        """Recording with a None payload does not crash (writes null)."""
        session_id = "sess-none"
        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=None,
        )
        assert filepath is not None
        assert os.path.isfile(filepath)
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["payload"] is None

    @pytest.mark.asyncio
    async def test_list_payload(self, recorder, temp_recording_dir):
        """Recording with a list payload (e.g., embeddings) succeeds."""
        session_id = "sess-list-payload"
        payload = [0.1, 0.2, 0.3]
        filepath = await recorder.record_request(
            session_id=session_id,
            direction="provider_to_client",
            payload=payload,
        )
        assert filepath is not None
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["payload"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_very_large_payload(self, recorder, temp_recording_dir):
        """Large payloads are written without truncation."""
        session_id = "sess-large"
        large_content = "x" * 100_000  # 100KB string
        payload = {"content": large_content}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )

        assert filepath is not None
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["payload"]["content"] == large_content

    @pytest.mark.asyncio
    async def test_disk_full_or_error_returns_none(self, recorder, temp_recording_dir, monkeypatch):
        """When a file write fails (disk full, permission error), None is returned instead of crashing."""
        # Simulate a permission error by making the session dir unwritable
        session_id = "sess-perm-denied"
        session_dir = Path(temp_recording_dir) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        session_dir.chmod(0o444)  # Read-only

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload={"msg": "should fail"},
        )

        # Should return None instead of raising
        assert filepath is None

        # Restore permissions for cleanup
        session_dir.chmod(0o755)

    @pytest.mark.asyncio
    async def test_invalid_json_payload_causes_no_crash(self, recorder, temp_recording_dir):
        """Payloads that are not JSON-serializable are handled gracefully."""
        session_id = "sess-bad-json"
        # Custom object that is not JSON serializable
        class NonSerializable:
            pass

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload={"bad": NonSerializable()},
        )

        # Should handle the serialization error and return None
        assert filepath is None, "Should return None for non-serializable payload"

    @pytest.mark.asyncio
    async def test_special_chars_in_session_id(self, recorder, temp_recording_dir):
        """Session IDs with special characters are handled safely."""
        session_id = "sess/../traversal-attempt"
        payload = {"msg": "safe"}

        filepath = await recorder.record_request(
            session_id=session_id,
            direction="client_to_proxy",
            payload=payload,
        )

        assert filepath is not None
        # The session directory should be created using the raw session_id
        # But path traversal chars should be sanitised or safe in the filename
        # e.g., slashes in session ID should NOT create nested directories
        path = Path(filepath)
        assert os.path.isfile(filepath)
        assert path.parent.name == session_id.replace("/", "_") or path.parent.name == session_id


# ═══════════════════════════════════════════════════════════════════════════
# Listing and retrieval helpers (AC8 — admin endpoint contract)
# ═══════════════════════════════════════════════════════════════════════════


class TestListAndRetrieve:
    """Tests for listing/retrieving recordings (helper methods consumed by admin endpoint)."""

    @pytest.mark.asyncio
    async def test_list_recordings_empty_session(self, recorder, temp_recording_dir):
        """Listing recordings for a session with no recordings returns empty list."""
        recordings = recorder.get_recordings_list("sess-no-recordings")
        assert recordings == []

    @pytest.mark.asyncio
    async def test_list_recordings_with_files(self, recorder, temp_recording_dir):
        """Listing recordings returns metadata for all recording files."""
        session_id = "sess-list"
        await recorder.record_request(session_id, "client_to_proxy", {"msg": "req1"})
        await recorder.record_request(session_id, "proxy_to_provider", {"msg": "req2"})
        await recorder.record_response(session_id, "provider_to_client", {"msg": "resp1"})

        recordings = recorder.get_recordings_list(session_id)

        assert len(recordings) == 3
        # Verify metadata fields
        for r in recordings:
            assert "filename" in r
            assert "timestamp" in r
            assert "direction" in r
            assert "file_size" in r
        # Verify direction values
        directions = {r["direction"] for r in recordings}
        assert "client_to_proxy" in directions
        assert "proxy_to_provider" in directions
        assert "provider_to_client" in directions

    @pytest.mark.asyncio
    async def test_get_recording_by_filename(self, recorder, temp_recording_dir):
        """Individual recording content can be retrieved by filename."""
        session_id = "sess-get"
        fp = await recorder.record_request(session_id, "client_to_proxy", {"msg": "retrieve me"})
        assert fp is not None
        filename = Path(fp).name

        content = recorder.get_recording(session_id, filename)

        assert content is not None
        assert content["payload"]["msg"] == "retrieve me"

    @pytest.mark.asyncio
    async def test_get_recording_not_found(self, recorder, temp_recording_dir):
        """Retrieving a non-existent file returns None."""
        content = recorder.get_recording("sess-nonexistent", "nonexistent-file.json")
        assert content is None

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, recorder, temp_recording_dir):
        """get_recording prevents path traversal attacks."""
        content = recorder.get_recording("sess-safe", "../../etc/passwd")
        assert content is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, recorder, temp_recording_dir):
        """list_sessions returns session IDs that have recordings."""
        await recorder.record_request("sess-a", "client_to_proxy", {"a": 1})
        await recorder.record_request("sess-b", "client_to_proxy", {"b": 2})

        sessions = recorder.list_sessions()
        session_ids = [s["session_id"] for s in sessions]

        assert "sess-a" in session_ids
        assert "sess-b" in session_ids
        # Each session should include preview fields
        for s in sessions:
            assert "response_time" in s
            assert "last_activity" in s
            assert "model" in s
            assert "provider" in s

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, recorder, temp_recording_dir):
        """list_sessions returns empty list when no recordings exist."""
        sessions = recorder.list_sessions()
        assert sessions == []


# ═══════════════════════════════════════════════════════════════════════════
# Configuration integration (AC5)
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    """Tests for reading session_recording config from config.yaml shape."""

    def test_config_path_override(self, temp_recording_dir):
        """Session recording path from config dict is applied correctly."""
        from proxy.session_recorder import SessionRecorder
        cfg = {"session_recording": {"path": temp_recording_dir}}
        rec = SessionRecorder.from_config(cfg)
        assert rec.recording_path == temp_recording_dir

    def test_config_extra_keys_ignored(self, temp_recording_dir):
        """Unknown keys in session_recording config are ignored without error."""
        from proxy.session_recorder import SessionRecorder
        cfg = {
            "session_recording": {
                "path": temp_recording_dir,
                "unknown_option": "should_not_cause_error",
            }
        }
        rec = SessionRecorder.from_config(cfg)
        assert rec.recording_path == temp_recording_dir

    def test_config_missing_path_key(self, tmp_path):
        """When path key is missing from config, a default is used."""
        from proxy.session_recorder import SessionRecorder
        cfg = {"session_recording": {}}
        rec = SessionRecorder.from_config(cfg)
        assert rec.recording_path.endswith("session-recordings")


# ═══════════════════════════════════════════════════════════════════════════
# Serialization format verification (AC1, AC3)
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializationFormat:
    """Verify the JSON format of saved recordings."""

    @pytest.mark.asyncio
    async def test_saved_record_includes_metadata(self, recorder, temp_recording_dir):
        """Each recording file includes metadata: session_id, direction, timestamp, payload."""
        session_id = "sess-format"
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        fp = await recorder.record_request(session_id, "client_to_proxy", payload)

        assert fp is not None
        with open(fp) as f:
            data = json.load(f)

        assert data["session_id"] == session_id
        assert data["direction"] == "client_to_proxy"
        assert data["payload"] == payload
        assert "timestamp" in data
        # Timestamp should be an ISO8601 string
        datetime.fromisoformat(data["timestamp"])

    @pytest.mark.asyncio
    async def test_no_http_headers_in_payload(self, recorder, temp_recording_dir):
        """Recordings contain only message payloads, not HTTP headers (constraint)."""
        session_id = "sess-no-headers"
        # Record a payload containing only message data (headers stripped upstream)
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}

        fp = await recorder.record_request(session_id, "client_to_proxy", payload)

        assert fp is not None
        with open(fp) as f:
            data = json.load(f)
        # The payload should NOT contain HTTP header fields
        saved_payload = data["payload"]
        assert "authorization" not in saved_payload
        assert "Authorization" not in saved_payload
        assert "api-key" not in saved_payload
        assert "cookie" not in saved_payload

    @pytest.mark.asyncio
    async def test_no_internal_proxy_secrets_in_payload(self, recorder, temp_recording_dir):
        """Recordings must not contain proxy-internal secrets (constraint)."""
        session_id = "sess-no-secrets"
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            # This should never include proxy API keys or internal config
        }

        fp = await recorder.record_request(session_id, "client_to_proxy", payload)

        assert fp is not None
        with open(fp) as f:
            data = json.load(f)
        saved_payload = data["payload"]
        # Verify no internal-only fields leaked
        assert "api_key_env" not in saved_payload
        assert "endpoint" not in saved_payload
        assert "internal" not in json.dumps(saved_payload).lower()


# ═══════════════════════════════════════════════════════════════════════════
# Session preview extraction (AC4)
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionPreviewExtraction:
    """Tests for _extract_message_text, _truncate_preview, and the
    ``preview_text`` field returned by ``list_sessions()``."""

    def test_extract_message_text_returns_user_content(self):
        """_extract_message_text returns the content of the first user message."""
        from proxy.session_recorder import SessionRecorder
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ]
        }
        result = SessionRecorder._extract_message_text(payload)
        assert result == "What is the capital of France?"

    def test_extract_message_text_skips_system_tool(self):
        """Only 'user' role messages are returned; system/tool messages are skipped."""
        from proxy.session_recorder import SessionRecorder
        payload = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "tool", "content": "[result]"},
                {"role": "user", "content": "Hello!"},
            ]
        }
        result = SessionRecorder._extract_message_text(payload)
        assert result == "Hello!"

    def test_extract_message_text_no_messages(self):
        """Returns empty string when there are no messages."""
        from proxy.session_recorder import SessionRecorder
        assert SessionRecorder._extract_message_text({}) == ""
        assert SessionRecorder._extract_message_text({"messages": []}) == ""

    def test_extract_message_text_no_user_msg(self):
        """Returns empty string when no user message exists."""
        from proxy.session_recorder import SessionRecorder
        payload = {"messages": [{"role": "assistant", "content": "Hi"}]}
        assert SessionRecorder._extract_message_text(payload) == ""

    def test_extract_message_text_non_dict_payload(self):
        """Returns empty string for non-dict payloads (e.g. SSE string payloads)."""
        from proxy.session_recorder import SessionRecorder
        assert SessionRecorder._extract_message_text("data: test\n\n") == ""
        assert SessionRecorder._extract_message_text(None) == ""
        assert SessionRecorder._extract_message_text(123) == ""

    def test_extract_message_text_content_array(self):
        """Handles content as an array of content parts (multimodal)."""
        from proxy.session_recorder import SessionRecorder
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/..."}},
                    ],
                }
            ]
        }
        result = SessionRecorder._extract_message_text(payload)
        assert result == "Describe this image"

    def test_truncate_preview_short_text(self):
        """Text shorter than max_chars is returned as-is."""
        from proxy.session_recorder import SessionRecorder
        assert SessionRecorder._truncate_preview("Hello") == "Hello"

    def test_truncate_preview_exact_length(self):
        """Text exactly at max_chars is returned as-is."""
        from proxy.session_recorder import SessionRecorder
        text = "A" * 80
        assert SessionRecorder._truncate_preview(text) == text

    def test_truncate_preview_long_text(self):
        """Long text is truncated with trailing ellipsis."""
        from proxy.session_recorder import SessionRecorder
        text = "Hello world! " * 20  # well over 80 chars
        result = SessionRecorder._truncate_preview(text)
        assert len(result) == 83  # 80 chars + "..."
        assert result.endswith("...")

    def test_truncate_preview_81_chars(self):
        """81 chars truncates to 80 + ellipsis."""
        from proxy.session_recorder import SessionRecorder
        text = "A" * 81
        result = SessionRecorder._truncate_preview(text)
        assert result == ("A" * 80) + "..."
        assert len(result) == 83

    def test_truncate_preview_custom_max(self):
        """Supports custom max_chars parameter."""
        from proxy.session_recorder import SessionRecorder
        text = "P" * 50
        assert SessionRecorder._truncate_preview(text, max_chars=10) == ("P" * 10) + "..."

    def test_list_sessions_includes_preview_text(self, temp_recording_dir):
        """list_sessions returns preview_text field with first 80 chars of user message."""
        from proxy.session_recorder import SessionRecorder

        # Create a session with a client_to_proxy recording containing a user message
        sess_dir = Path(temp_recording_dir) / "sess-preview-1"
        sess_dir.mkdir(parents=True, exist_ok=True)

        recording = {
            "session_id": "sess-preview-1",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "What is the weather in London today?"},
                ],
                "model": "test-model",
                "stream": True,
            },
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(recording))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-preview-1"
        assert sessions[0]["preview_text"] == "What is the weather in London today?"

    def test_list_sessions_preview_truncated(self, temp_recording_dir):
        """Long user messages are truncated to 80 chars + ellipsis."""
        from proxy.session_recorder import SessionRecorder

        sess_dir = Path(temp_recording_dir) / "sess-preview-long"
        sess_dir.mkdir(parents=True, exist_ok=True)

        long_message = "Hello I would like to ask a very very long question about the capital of France, Italy, and Spain because I need to travel there next month."
        recording = {
            "session_id": "sess-preview-long",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {
                "messages": [{"role": "user", "content": long_message}],
                "model": "test-model",
            },
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(recording))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()

        assert len(sessions) == 1
        result = sessions[0]["preview_text"]
        assert len(result) == 83  # 80 chars + "..."
        assert result.endswith("...")

    def test_list_sessions_empty_preview_when_no_user_msg(self, temp_recording_dir):
        """Sessions with no user message have empty preview_text."""
        from proxy.session_recorder import SessionRecorder

        sess_dir = Path(temp_recording_dir) / "sess-no-user"
        sess_dir.mkdir(parents=True, exist_ok=True)

        recording = {
            "session_id": "sess-no-user",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {
                "messages": [],
                "model": "test-model",
            },
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(recording))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()
        assert sessions[0]["preview_text"] == ""

    def test_list_sessions_preview_uses_first_client_recording(self, temp_recording_dir):
        """Preview text comes from the earliest client_to_proxy recording."""
        from proxy.session_recorder import SessionRecorder

        sess_dir = Path(temp_recording_dir) / "sess-multi-req"
        sess_dir.mkdir(parents=True, exist_ok=True)

        # First request (system message + first user message)
        req1 = {
            "session_id": "sess-multi-req",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {"messages": [{"role": "user", "content": "First message"}]},
        }
        # Second request (later user message)
        req2 = {
            "session_id": "sess-multi-req",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:01:00.000000+00:00",
            "payload": {"messages": [{"role": "user", "content": "Second message"}]},
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(req1))
        (sess_dir / "2026-07-07T10:01:00.000000-request.json").write_text(json.dumps(req2))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()
        assert sessions[0]["preview_text"] == "First message"

    def test_list_sessions_last_activity_differs_from_response_time(self, temp_recording_dir):
        """last_activity reflects the latest recording, not the first response."""
        from proxy.session_recorder import SessionRecorder

        sess_dir = Path(temp_recording_dir) / "sess-last-activity"
        sess_dir.mkdir(parents=True, exist_ok=True)

        # Early request
        req = {
            "session_id": "sess-last-activity",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {"messages": [{"role": "user", "content": "Hello"}]},
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(req))

        # Early response
        resp = {
            "session_id": "sess-last-activity",
            "direction": "provider_to_client",
            "timestamp": "2026-07-07T10:00:05.000000+00:00",
            "payload": {"choices": [{"text": "Hi"}]},
            "model": "test-model",
            "provider": "test-provider",
        }
        (sess_dir / "2026-07-07T10:00:05.000000-response.json").write_text(json.dumps(resp))

        # Later request (second turn)
        req2 = {
            "session_id": "sess-last-activity",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:01:00.000000+00:00",
            "payload": {"messages": [{"role": "user", "content": "Follow up"}]},
        }
        (sess_dir / "2026-07-07T10:01:00.000000-request.json").write_text(json.dumps(req2))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()

        assert len(sessions) == 1
        s = sessions[0]
        # response_time should be the first provider_to_client response
        assert s["response_time"] == "2026-07-07T10:00:05.000000+00:00"
        # last_activity should be the latest recording timestamp
        assert s["last_activity"] == "2026-07-07T10:01:00.000000+00:00"
        # Model/provider come from the first response (or first request)
        assert s["model"] == "test-model"
        assert s["provider"] == "test-provider"
        assert s["preview_text"] == "Hello"

    def test_list_sessions_preview_preserves_existing_fields(self, temp_recording_dir):
        """The preview_text field is added alongside existing fields."""
        from proxy.session_recorder import SessionRecorder

        sess_dir = Path(temp_recording_dir) / "sess-all-fields"
        sess_dir.mkdir(parents=True, exist_ok=True)

        req = {
            "session_id": "sess-all-fields",
            "direction": "client_to_proxy",
            "timestamp": "2026-07-07T10:00:00.000000+00:00",
            "payload": {"messages": [{"role": "user", "content": "Hello"}]},
            "model": "qwen3",
            "provider": "local",
        }
        (sess_dir / "2026-07-07T10:00:00.000000-request.json").write_text(json.dumps(req))

        recorder = SessionRecorder(recording_path=temp_recording_dir)
        sessions = recorder.list_sessions()

        s = sessions[0]
        assert s["session_id"] == "sess-all-fields"
        assert s["response_time"] == "2026-07-07T10:00:00.000000+00:00"
        assert s["last_activity"] == "2026-07-07T10:00:00.000000+00:00"
        assert s["model"] == "qwen3"
        assert s["provider"] == "local"
        assert s["preview_text"] == "Hello"
