"""
Tests for session_slot_pool_size / --parallel alignment and slot save/restore.

Verifies that:
- _slot_id_for_session returns consistent results within the pool size
- _build_slot_context returns correct slot_id, filename, timeout from config
- slot save/restore functions correctly handle the slot_id range
- LLAMA_PARALLEL env var is forwarded from lifecycle to start script
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.session import (
    _slot_id_for_session,
    _build_slot_context,
    _slot_persistence_enabled,
    _slot_filename_for_session,
    _save_slot_snapshot,
    _restore_slot_snapshot,
)


# ===================================================================
# _slot_persistence_enabled
# ===================================================================


class TestSlotPersistenceEnabled:
    """Tests for the slot persistence gate."""

    def test_path_and_pool_size_positive(self):
        """Both slot_path and slot_pool_size > 0 → True."""
        assert _slot_persistence_enabled("/tmp/slots", 2) is True

    def test_path_none_returns_false(self):
        """None slot_path → False."""
        assert _slot_persistence_enabled(None, 2) is False

    def test_pool_size_zero_returns_false(self):
        """slot_pool_size == 0 → False."""
        assert _slot_persistence_enabled("/tmp/slots", 0) is False

    def test_pool_size_negative_returns_false(self):
        """slot_pool_size < 0 → False (bool(-1) is True but our function checks > 0)."""
        assert _slot_persistence_enabled("/tmp/slots", -1) is False


# ===================================================================
# _slot_id_for_session
# ===================================================================


class TestSlotIdForSession:
    """Verifies slot ID is deterministic and within pool bounds."""

    def test_deterministic_across_calls(self):
        """Same session_id + pool_size produces same slot_id every time."""
        sid_a = _slot_id_for_session("session-foo", 4)
        sid_b = _slot_id_for_session("session-foo", 4)
        assert sid_a == sid_b
        assert 0 <= sid_a < 4

    def test_different_sessions_may_map_to_different_slots(self):
        """Different session IDs may produce different slot IDs."""
        sid1 = _slot_id_for_session("session-alpha", 4)
        sid2 = _slot_id_for_session("session-beta", 4)
        # They could collide by chance, but should be valid
        assert 0 <= sid1 < 4
        assert 0 <= sid2 < 4

    def test_pool_size_one(self):
        """pool_size=1 always returns slot_id 0."""
        sid = _slot_id_for_session("anything", 1)
        assert sid == 0

    def test_pool_size_matches_session_distribution(self):
        """With pool_size=N, all returned IDs are in [0, N-1]."""
        for pool in [1, 2, 3, 4, 8]:
            for session in [f"session-{i}" for i in range(100)]:
                sid = _slot_id_for_session(session, pool)
                assert 0 <= sid < pool, (
                    f"session={session} pool={pool} got sid={sid}"
                )

    def test_empty_session_id_returns_none(self):
        """Empty session_id returns None."""
        assert _slot_id_for_session("", 4) is None

    def test_pool_size_zero_returns_none(self):
        """pool_size of zero returns None."""
        assert _slot_id_for_session("test", 0) is None


# ===================================================================
# _build_slot_context
# ===================================================================


class TestBuildSlotContext:
    """Verify _build_slot_context extracts correct values from config."""

    def test_valid_config_returns_slot_context(self):
        """With slot_path + slot_pool_size > 0 returns (slot_id, filename, timeout)."""
        config = {
            "session_slot_save_path": "/tmp/slot-cache",
            "session_slot_pool_size": 4,
            "session_slot_timeout_seconds": 5.0,
        }
        slot_id, filename, timeout = _build_slot_context(config, "test-session")
        assert slot_id is not None
        assert 0 <= slot_id < 4
        assert filename is not None
        assert "slot_" in filename
        assert timeout == 5.0

    def test_no_slot_path_returns_none(self):
        """Without slot_path, returns None entries."""
        config = {
            "session_slot_pool_size": 4,
        }
        slot_id, filename, timeout = _build_slot_context(config, "test-session")
        assert slot_id is None
        assert filename is None

    def test_pool_size_zero_returns_none(self):
        """pool_size=0 returns None entries."""
        config = {
            "session_slot_save_path": "/tmp/slots",
            "session_slot_pool_size": 0,
        }
        slot_id, filename, timeout = _build_slot_context(config, "test-session")
        assert slot_id is None
        assert filename is None

    def test_no_session_id_returns_none(self):
        """No session_id returns None entries."""
        config = {
            "session_slot_save_path": "/tmp/slots",
            "session_slot_pool_size": 4,
        }
        slot_id, filename, timeout = _build_slot_context(config, None)
        assert slot_id is None
        assert filename is None

    def test_default_timeout(self):
        """When timeout is missing, returns default 3.0."""
        config = {
            "session_slot_save_path": "/tmp/slots",
            "session_slot_pool_size": 2,
        }
        _, _, timeout = _build_slot_context(config, "test-session")
        assert timeout == 3.0


# ===================================================================
# _save_slot_snapshot / _restore_slot_snapshot (with mock backend)
# ===================================================================


class TestSlotSaveRestoreMocked:
    """Verify slot save/restore functions call the HTTP endpoint correctly."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_srv = MagicMock()
        self.mock_srv.logger = MagicMock()
        self.mock_srv._http_client = None
        patcher = patch("proxy.session._srv", return_value=self.mock_srv)
        patcher.start()
        yield
        patcher.stop()

    @pytest.mark.asyncio
    async def test_save_correct_slot_id(self):
        """_save_slot_snapshot calls /slots/<id>?action=save with correct slot_id."""
        with patch("proxy.session.httpx.AsyncClient") as mock_client_cls:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await _save_slot_snapshot(
                llama_port=8080,
                slot_id=2,
                filename="/tmp/slots/test.bin",
                timeout=3.0,
            )

            assert result is True
            # Verify the URL contains slot_id=2
            call_url = mock_client.post.call_args[0][0]
            assert "slots/2" in call_url
            assert "action=save" in call_url

    @pytest.mark.asyncio
    async def test_restore_correct_slot_id(self):
        """_restore_slot_snapshot calls /slots/<id>?action=restore."""
        with (
            patch("proxy.session.Path.exists", return_value=True),
            patch("proxy.session.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await _restore_slot_snapshot(
                llama_port=8080,
                slot_id=3,
                filename="/tmp/slots/test.bin",
                timeout=3.0,
            )

            assert result is True
            call_url = mock_client.post.call_args[0][0]
            assert "slots/3" in call_url
            assert "action=restore" in call_url

    @pytest.mark.asyncio
    async def test_restore_missing_file_skips(self):
        """_restore_slot_snapshot returns False without HTTP call if file missing."""
        with patch("proxy.session.Path.exists", return_value=False):
            result = await _restore_slot_snapshot(
                llama_port=8080,
                slot_id=0,
                filename="/tmp/slots/nonexistent.bin",
                timeout=3.0,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_save_with_model_payload(self):
        """_save_slot_snapshot passes model in payload when provided."""
        with patch("proxy.session.httpx.AsyncClient") as mock_client_cls:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await _save_slot_snapshot(
                llama_port=8080,
                slot_id=1,
                filename="/tmp/slots/test.bin",
                timeout=3.0,
                model="qwen3",
            )

            assert result is True
            call_kwargs = mock_client.post.call_args[1]
            assert call_kwargs["json"]["model"] == "qwen3"
            assert call_kwargs["json"]["filename"] == "test.bin"


# ===================================================================
# _slot_filename_for_session
# ===================================================================


class TestSlotFilenameForSession:
    """Verify slot filenames are deterministic and safe."""

    def test_filename_contains_session_id(self):
        """Filename includes session_id (sanitized)."""
        fname = _slot_filename_for_session("my-session-123", "/tmp/slots")
        assert "slot_my-session-123" in fname
        assert fname.startswith("/tmp/slots/")

    def test_special_chars_sanitized(self):
        """Special characters in session_id are replaced."""
        fname = _slot_filename_for_session("bad/name:chars", "/tmp/slots")
        assert "slot_bad_name_chars" in fname

    def test_empty_session_id(self):
        """Empty session_id produces slot_.bin."""
        fname = _slot_filename_for_session("", "/tmp/slots")
        assert fname == "/tmp/slots/slot_.bin"


# ===================================================================
# Integration: LLAMA_PARALLEL env var is exported by lifecycle
# ===================================================================


class TestLifecycleExportsLLamaParallel:
    """Verify the lifecycle sets LLAMA_PARALLEL based on config."""

    def test_lifecycle_sets_llama_parallel(self):
        """lifecycle.start_llama_server exports LLAMA_PARALLEL = session_slot_pool_size."""
        from proxy.lifecycle import start_llama_server

        mock_config = {
            "llama_server_port": 8080,
            "llama_start_script": "/bin/echo",
            "session_slot_save_path": "/tmp/slots",
            "session_slot_pool_size": 2,
        }

        mock_srv = MagicMock()
        mock_srv.config = {"server": mock_config}
        mock_srv.logger = MagicMock()
        mock_srv.log_dir = None
        mock_srv.last_start_failure = None
        mock_srv.broadcast_status_sync = MagicMock()

        with (
            patch("proxy.lifecycle._srv", return_value=mock_srv),
            patch("proxy.lifecycle.spawn_and_capture") as mock_spawn,
        ):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_spawn.return_value = (mock_proc, "started")

            # Call the function with router mode
            start_llama_server(model=None)

            # Verify LLAMA_PARALLEL was set in the env passed to spawn_and_capture
            env_captured = mock_spawn.call_args[0][1]  # env is 2nd positional arg
            assert env_captured.get("LLAMA_PARALLEL") == "2", (
                f"Expected LLAMA_PARALLEL=2, got {env_captured.get('LLAMA_PARALLEL')}"
            )

    def test_lifecycle_defaults_to_one(self):
        """When slot_pool_size is not set, LLAMA_PARALLEL defaults to 1."""
        from proxy.lifecycle import start_llama_server

        mock_config = {
            "llama_server_port": 8080,
            "llama_start_script": "/bin/echo",
        }

        mock_srv = MagicMock()
        mock_srv.config = {"server": mock_config}
        mock_srv.logger = MagicMock()
        mock_srv.log_dir = None
        mock_srv.last_start_failure = None
        mock_srv.broadcast_status_sync = MagicMock()

        with (
            patch("proxy.lifecycle._srv", return_value=mock_srv),
            patch("proxy.lifecycle.spawn_and_capture") as mock_spawn,
        ):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_spawn.return_value = (mock_proc, "started")

            start_llama_server(model=None)

            env_captured = mock_spawn.call_args[0][1]
            assert env_captured.get("LLAMA_PARALLEL") == "1"


# ===================================================================
# Integration: _slot_id_for_session consistency with pool_size
# ===================================================================


class TestSlotDistributionConsistency:
    """Verify slot distribution is consistent and valid for all pool sizes."""

    def test_pool_size_two_coverage(self):
        """With pool_size=2, at least some sessions map to each slot."""
        slot0_count = 0
        slot1_count = 0
        for i in range(100):
            sid = _slot_id_for_session(f"session-{i}", 2)
            if sid == 0:
                slot0_count += 1
            else:
                slot1_count += 1
        assert slot0_count > 0
        assert slot1_count > 0

    def test_pool_size_equals_parallel(self):
        """Simulate alignment: pool_size=N matches --parallel N → all IDs valid."""
        for pool_size in [1, 2, 4]:
            for i in range(50):
                sid = _slot_id_for_session(f"test-{i}", pool_size)
                assert 0 <= sid < pool_size
