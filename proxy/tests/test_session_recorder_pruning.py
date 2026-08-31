"""Tests for session-recording retention pruning (LP-0MT2TC3PB005H1RD).

Covers:
- Old recordings pruned, fresh recordings retained (3-day retention default)
- Empty session directories cleaned up after pruning
- Retention configured via ``session_recording.retention_days`` (default 3,
  0 = disabled)
- Path safety: pruning never leaves the configured root and never follows
  symlinks
- Pruning uses embedded timestamps (filename, then content), not mtime
- Startup prune runs as a background task and never blocks startup
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _iso(offset_days: float = 0.0) -> str:
    """ISO filename timestamp *offset_days* from now (UTC, microsecond fmt)."""
    return (datetime.now(UTC) + timedelta(days=offset_days)).isoformat(
        timespec="microseconds"
    )


def _write_recording(session_dir: Path, iso_ts: str, suffix: str = "request",
                     content_timestamp: str | None = None) -> Path:
    """Write a recording file named by *iso_ts* into *session_dir*."""
    fname = f"{iso_ts}-{suffix}.json"
    path = session_dir / fname
    payload = {
        "session_id": session_dir.name,
        "direction": "client_to_proxy",
        "timestamp": content_timestamp or iso_ts,
        "payload": {"messages": [{"role": "user", "content": "hello"}]},
    }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def temp_recording_dir(tmp_path):
    """Provide a temporary root recording directory."""
    d = tmp_path / "session-recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def recorder(temp_recording_dir):
    """Return a SessionRecorder configured with a temp directory."""
    from proxy.session_recorder import SessionRecorder
    return SessionRecorder(recording_path=str(temp_recording_dir))


class TestPruning:
    """Pruning deletes only recordings older than the retention window."""

    @pytest.mark.asyncio
    async def test_prune_removes_old_files_and_keeps_fresh(self, recorder, temp_recording_dir):
        """Files older than retention are deleted; fresh ones survive."""
        sess_dir = temp_recording_dir / "sess-old-fresh"
        sess_dir.mkdir()
        _write_recording(sess_dir, _iso(-10))    # 10 days old
        fresh = _write_recording(sess_dir, _iso(-1))  # 1 day old

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert not fresh.parent.joinpath(f"{_iso(-10)}-request.json").exists()
        assert fresh.exists()

    @pytest.mark.asyncio
    async def test_prune_removes_empty_session_dirs(self, recorder, temp_recording_dir):
        """A session dir whose only recordings were pruned is removed."""
        sess_dir = temp_recording_dir / "sess-all-old"
        sess_dir.mkdir()
        _write_recording(sess_dir, _iso(-10))

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert stats["dirs_removed"] == 1
        assert not sess_dir.exists()

    @pytest.mark.asyncio
    async def test_prune_keeps_dir_with_fresh_recordings(self, recorder, temp_recording_dir):
        """A dir with fresh recordings is kept even if old files were pruned."""
        sess_dir = temp_recording_dir / "sess-kept"
        sess_dir.mkdir()
        _write_recording(sess_dir, _iso(-10))
        fresh = _write_recording(sess_dir, _iso(0))

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert stats["dirs_removed"] == 0
        assert sess_dir.exists()
        assert fresh.exists()

    @pytest.mark.asyncio
    async def test_prune_handles_naive_timestamps(self, recorder, temp_recording_dir):
        """Legacy recordings with naive ISO timestamps are pruned correctly.

        Older recordings embedded timestamps without a tz offset; pruning
        must treat them as UTC and compare safely against the cutoff.
        """
        sess_dir = temp_recording_dir / "sess-naive"
        sess_dir.mkdir()
        old_naive = (datetime.now(UTC) - timedelta(days=10)).replace(tzinfo=None)
        fresh_naive = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None)
        old_path = _write_recording(
            sess_dir, old_naive.isoformat(timespec="microseconds"),
            content_timestamp=old_naive.isoformat(timespec="microseconds"),
        )
        fresh_path = _write_recording(
            sess_dir, fresh_naive.isoformat(timespec="microseconds"),
            content_timestamp=fresh_naive.isoformat(timespec="microseconds"),
        )

        stats = await recorder.prune_older_than(3)

        assert stats["errors"] == 0
        assert stats["files_deleted"] == 1
        assert not old_path.exists()
        assert fresh_path.exists()

    @pytest.mark.asyncio
    async def test_prune_uses_filename_timestamp_not_mtime(self, recorder, temp_recording_dir):
        """Age is judged by the embedded timestamp, not file mtime.

        A file written now but named with an old timestamp is pruned;
        a file written long ago but named fresh is retained.
        """
        sess_dir = temp_recording_dir / "sess-ts-rule"
        sess_dir.mkdir()
        # Old filename timestamp, brand-new mtime → still pruned.
        old_named = _write_recording(sess_dir, _iso(-10))
        # New filename timestamp → kept regardless of mtime.
        fresh_named = _write_recording(sess_dir, _iso(0))

        old_mtime = datetime.now().timestamp() + 60
        import os
        os.utime(old_named, (old_mtime, old_mtime))
        fresh_mtime = datetime.now().timestamp() - 10 * 86400
        os.utime(fresh_named, (fresh_mtime, fresh_mtime))

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert not old_named.exists()
        assert fresh_named.exists()

    @pytest.mark.asyncio
    async def test_prune_falls_back_to_content_timestamp(self, recorder, temp_recording_dir):
        """Files without a parseable filename timestamp fall back to content."""
        sess_dir = temp_recording_dir / "sess-content-ts"
        sess_dir.mkdir()
        # Unparseable filename → content timestamp (10 days old) decides.
        fname = "unparseable-name-request.json"
        path = sess_dir / fname
        path.write_text(json.dumps({
            "session_id": "sess-content-ts",
            "direction": "client_to_proxy",
            "timestamp": _iso(-10),
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
        }))

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_prune_does_not_follow_symlink_out_of_root(self, recorder, temp_recording_dir):
        """Pruning must not delete files reachable via symlink out of the root."""
        outside = temp_recording_dir.parent / "outside"
        outside.mkdir()
        victim = outside / "victim.json"
        victim.write_text("{}")

        link_dir = temp_recording_dir / "sess-link"
        link_dir.symlink_to(outside, target_is_directory=True)

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 0
        assert victim.exists()

    @pytest.mark.asyncio
    async def test_prune_ignores_non_json_non_recording_files(self, recorder, temp_recording_dir):
        """Only recording files (…-request.json/…-response.json) are pruned."""
        sess_dir = temp_recording_dir / "sess-mixed"
        sess_dir.mkdir()
        _write_recording(sess_dir, _iso(-10))
        scratch = sess_dir / "scratch.txt"
        scratch.write_text("keep me")

        stats = await recorder.prune_older_than(3)

        assert stats["files_deleted"] == 1
        assert scratch.exists()


class TestPruningConfig:
    """Retention configuration: default, override, and disable."""

    def test_retention_days_default(self, temp_recording_dir):
        """Default retention is 3 days."""
        from proxy.session_recorder import DEFAULT_RETENTION_DAYS, SessionRecorder
        rec = SessionRecorder(recording_path=str(temp_recording_dir))
        assert rec.retention_days == DEFAULT_RETENTION_DAYS
        assert DEFAULT_RETENTION_DAYS == 3

    def test_retention_days_from_config(self, temp_recording_dir):
        """retention_days is read from session_recording config."""
        from proxy.session_recorder import SessionRecorder

        cfg = {"session_recording": {
            "path": str(temp_recording_dir),
            "retention_days": 7,
        }}
        rec = SessionRecorder.from_config(cfg)
        assert rec.retention_days == 7

    @pytest.mark.asyncio
    async def test_prune_disabled_retains_everything(self, recorder, temp_recording_dir):
        """retention_days <= 0 disables pruning: nothing is deleted."""
        recorder.retention_days = 0
        sess_dir = temp_recording_dir / "sess-keep"
        sess_dir.mkdir()
        old = _write_recording(sess_dir, _iso(-30))

        stats = await recorder.prune_older_than(recorder.retention_days)

        assert stats["disabled"] is True
        assert stats["files_deleted"] == 0
        assert old.exists()

    @pytest.mark.asyncio
    async def test_prune_returns_stats(self, recorder, temp_recording_dir):
        """Prune returns a stats dict with counts."""
        sess_dir = temp_recording_dir / "sess-stats"
        sess_dir.mkdir()
        _write_recording(sess_dir, _iso(-10))

        stats = await recorder.prune_older_than(3)

        assert stats["files_scanned"] >= 1
        assert stats["files_deleted"] == 1
        assert stats["dirs_removed"] == 1
        assert stats["errors"] == 0


class TestPruningPathSafety:
    """Pruning never leaves the configured recording root."""

    @pytest.mark.asyncio
    async def test_prune_ignores_unknown_subdirs_without_escape(self, recorder, temp_recording_dir):
        """Non-recording subdirs are scanned but nothing outside root is touched."""
        outside = temp_recording_dir.parent / "outside"
        outside.mkdir()
        victim = outside / "victim.json"
        victim.write_text("{}")

        # A session-id-like dir with a traversal name must not escape.
        evil = temp_recording_dir / ".."
        if not evil.exists():
            evil.mkdir()
        probe = temp_recording_dir / "sess-probe"
        probe.mkdir()
        _write_recording(probe, _iso(-10))

        stats = await recorder.prune_older_than(3)

        assert stats["errors"] == 0
        assert victim.exists()


class TestPruneServerIntegration:
    """Startup prune task wiring (non-blocking background task)."""

    @pytest.mark.asyncio
    async def test_startup_prune_task_cancel_cleanly(self, recorder):
        """The startup prune task can be cancelled without pending work."""
        import proxy.server as server

        with patch.object(server, "config", {"session_recording": {"path": recorder.recording_path}}):
            server._startup_launch_recording_prune()
            try:
                assert server._recording_prune_task is not None
                assert not server._recording_prune_task.done()
            finally:
                server._shutdown_stop_recording_prune()
                assert server._recording_prune_task is None
