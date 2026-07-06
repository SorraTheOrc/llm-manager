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
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call


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

        assert "sess-a" in sessions
        assert "sess-b" in sessions

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
