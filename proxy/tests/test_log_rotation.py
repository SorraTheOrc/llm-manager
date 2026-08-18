"""Tests for log rotation / pruning (LP-0MSNKMXIK004P7TL).

Tests cover:
- prune_logs deletes files older than retention_days
- prune_logs respects the retention boundary (keeps recent files)
- prune_logs handles compressed (.gz) rotated files
- setup_logging reads retention_days from config (default 7 days)
- Pruning is called on startup and logs the count
"""

import gzip
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# prune_logs unit tests
# ---------------------------------------------------------------------------


def test_prune_logs_deletes_old_files(tmp_path):
    """Files older than retention_days should be deleted (LP-0MSNKMXIK004P7TL AC1)."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create an old file (10 days ago)
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d_%H")
    old_file = log_dir / f"proxy.log.{old_date}"
    old_file.write_text("old log data\n")

    # Create a recent file (1 day ago)
    recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d_%H")
    recent_file = log_dir / f"proxy.log.{recent_date}"
    recent_file.write_text("recent log data\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)

    assert deleted == 1, f"Expected 1 file deleted, got {deleted}"
    assert not old_file.exists(), "Old file should have been deleted"
    assert recent_file.exists(), "Recent file should still exist"


def test_prune_logs_respects_boundary(tmp_path):
    """Files older than exactly retention_days should be deleted; files at the
    boundary (>= cutoff comparison makes exactly-7-days-old files still
    deletable if they are < cutoff by time-of-day) — this test uses files
    clearly inside retention."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # File just inside retention (6 days ago) — within retention, but older
    # than the 1-day compress threshold, so it gets compressed.
    inside_date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d_%H")
    inside_file = log_dir / f"proxy.log.{inside_date}"
    inside_file.write_text("inside retention\n")

    # File clearly outside retention (15 days ago)
    outside_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d_%H")
    outside_file = log_dir / f"proxy.log.{outside_date}"
    outside_file.write_text("outside retention\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)

    assert deleted == 1, f"Expected 1 file deleted, got {deleted}"
    assert not outside_file.exists(), "File outside retention should be deleted"
    # Inside-retention file is retained as a compressed .gz (AC2)
    gz_file = log_dir / f"proxy.log.{inside_date}.gz"
    assert gz_file.exists(), "File inside retention should be retained (compressed)"


def test_prune_logs_handles_gzipped_files(tmp_path):
    """Compressed (.gz) rotated files are also pruned."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create an old gzipped file (10 days ago)
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d_%H")
    old_file = log_dir / f"proxy.log.{old_date}.gz"
    with gzip.open(old_file, "wt") as f:
        f.write("old compressed data\n")

    # Create a recent gzipped file (2 days ago)
    recent_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d_%H")
    recent_file = log_dir / f"proxy.log.{recent_date}.gz"
    with gzip.open(recent_file, "wt") as f:
        f.write("recent compressed data\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)

    assert deleted == 1, f"Expected 1 gzipped file deleted, got {deleted}"
    assert not old_file.exists()
    assert recent_file.exists()


def test_prune_logs_compresses_old_retained_files(tmp_path):
    """Files within retention but older than compress_after_days are gzipped,
    and the plain-text file is removed (LP-0MSNKMXIK004P7TL AC2)."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # A file 3 days old — within 7-day retention, but older than 1-day
    # compress threshold, so it should be compressed.
    old_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d_%H")
    plain_file = log_dir / f"proxy.log.{old_date}"
    plain_file.write_text("data to compress\n" * 100)

    # A file from today — within compression threshold, stays plain
    today = datetime.now().strftime("%Y-%m-%d_%H")
    fresh_file = log_dir / f"proxy.log.{today}"
    fresh_file.write_text("fresh data\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)

    assert deleted == 0, "No files should be deleted (all within retention)"

    gz_name = f"proxy.log.{old_date}.gz"
    assert not plain_file.exists(), "Plain file should have been replaced by .gz"
    gz_file = log_dir / gz_name
    assert gz_file.exists(), f"Expected gzipped file {gz_name}"
    with gzip.open(gz_file, "rt") as f:
        content = f.read()
    assert "data to compress" in content, "Compressed content should be readable"
    assert fresh_file.exists(), "Fresh file should remain uncompressed"


def test_prune_logs_does_not_double_compress(tmp_path):
    """Running prune twice should not re-compress an already-.gz file."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    old_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d_%H")
    plain_file = log_dir / f"proxy.log.{old_date}"
    plain_file.write_text("compress me\n" * 50)

    utils_mod.prune_logs(log_dir, retention_days=7)
    gz_file = log_dir / f"proxy.log.{old_date}.gz"
    assert gz_file.exists()
    gz_size = gz_file.stat().st_size

    # Second run: no plain file to compress, nothing should change
    deleted = utils_mod.prune_logs(log_dir, retention_days=7)
    assert deleted == 0
    assert gz_file.exists()
    assert gz_file.stat().st_size == gz_size, "Should not re-compress"


def test_compress_after_days_disabled(tmp_path):
    """compress_after_days=0 disables compression entirely."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    old_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d_%H")
    plain_file = log_dir / f"proxy.log.{old_date}"
    plain_file.write_text("keep plain\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7, compress_after_days=0)
    assert deleted == 0
    assert plain_file.exists(), "File should stay uncompressed when compression is off"
    assert not (log_dir / f"proxy.log.{old_date}.gz").exists()


def test_prune_logs_skips_unrecognized_filenames(tmp_path):
    """Files that don't match the proxy.log.YYYY-MM-DD_HH pattern are skipped."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create a file with an unrecognized pattern
    weird_file = log_dir / "proxy.log.1"
    weird_file.write_text("weird filename\n")

    # Create a llama-server log (should be skipped)
    server_file = log_dir / "llama-server.5.log"
    server_file.write_text("server log\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)

    assert deleted == 0
    assert weird_file.exists()
    assert server_file.exists()


def test_prune_logs_nonexistent_dir(tmp_path):
    """prune_logs returns 0 when the log directory doesn't exist."""
    from proxy import utils as utils_mod

    deleted = utils_mod.prune_logs(tmp_path / "nonexistent", retention_days=7)
    assert deleted == 0


def test_prune_logs_empty_dir(tmp_path):
    """prune_logs returns 0 when there are no proxy.log.* files."""
    from proxy import utils as utils_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create a non-matching file
    (log_dir / "other.log").write_text("other\n")

    deleted = utils_mod.prune_logs(log_dir, retention_days=7)
    assert deleted == 0


# ---------------------------------------------------------------------------
# setup_logging config reading tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_handlers():
    """Remove existing llama-proxy handlers so each test starts fresh."""
    logger = logging.getLogger("llama-proxy")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    yield
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_setup_logging_reads_retention_days_from_config(tmp_path, monkeypatch, _clean_handlers):
    """setup_logging uses retention_days from config to compute backupCount."""
    from proxy.server import setup_logging

    log_dir = str(tmp_path / "proxy-logs")
    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)

    # Explicit config value
    config = {
        "logging": {
            "directory": log_dir,
            "retention_days": 14,
            "rotation_hours": 6,
            "level": "INFO",
        }
    }

    logger = setup_logging(config)

    # The backupCount should be (14 * 24) // 6 = 56
    handlers = [
        h for h in logger.handlers
        if hasattr(h, "backupCount")
    ]
    assert len(handlers) == 1
    assert handlers[0].backupCount == 56, (
        f"Expected backupCount=56 (14 days * 24 / 6), got {handlers[0].backupCount}"
    )


def test_setup_logging_default_retention_is_7(tmp_path, monkeypatch, _clean_handlers):
    """Default retention_days should be 7 (not 90)."""
    from proxy.server import setup_logging

    log_dir = str(tmp_path / "proxy-logs")
    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)

    # Config without retention_days — should use default of 7
    config = {
        "logging": {
            "directory": log_dir,
            "rotation_hours": 6,
            "level": "INFO",
        }
    }

    logger = setup_logging(config)

    handlers = [
        h for h in logger.handlers
        if hasattr(h, "backupCount")
    ]
    assert len(handlers) == 1
    assert handlers[0].backupCount == 28, (
        f"Expected default backupCount=28 (7 days * 24 / 6), got {handlers[0].backupCount}"
    )


def test_setup_logging_pruning_logs_on_startup(tmp_path, monkeypatch, caplog, _clean_handlers):
    """Pruning runs on startup and logs how many files were deleted."""
    from proxy.server import setup_logging

    log_dir = tmp_path / "proxy-logs"
    log_dir.mkdir(parents=True)
    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)

    # Create old log files
    for days_ago in range(10, 15):
        old_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d_%H")
        (log_dir / f"proxy.log.{old_date}").write_text("old data\n")

    config = {
        "logging": {
            "directory": str(log_dir),
            "retention_days": 7,
            "rotation_hours": 6,
            "level": "INFO",
        }
    }

    setup_logging(config)

    # Check that pruning was logged
    assert any(
        "Pruned" in record.getMessage() and "old rotated log" in record.getMessage().lower()
        or "pruned" in record.getMessage().lower()
        for record in caplog.records
    ), "Expected prune log message"

    # Verify files were actually deleted (all 5 old files are 10-14 days old)
    remaining = list(log_dir.glob("proxy.log.*"))
    assert len(remaining) == 0, (
        f"Expected all old files deleted, but {len(remaining)} remain"
    )


def test_config_files_have_7_day_retention():
    """All config files should have retention_days: 7 (LP-0MSNKMXIK004P7TL AC5)."""
    import yaml

    proxy_root = Path(__file__).resolve().parent.parent  # proxy/
    config_files = [
        proxy_root / "config.yaml",
        proxy_root / "config-cheap.yaml",
        proxy_root / "config-fast.yaml",
    ]

    for config_path in config_files:
        assert config_path.exists(), f"Config file missing: {config_path}"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        retention = cfg.get("logging", {}).get("retention_days")
        assert retention == 7, (
            f"{config_path.name}: expected retention_days=7, got {retention}"
        )


def test_logrotate_dropin_exists():
    """The logrotate config file should exist in scripts/ (LP-0MSNKMXIK004P7TL AC3)."""
    proxy_root = Path(__file__).resolve().parent.parent  # proxy/
    dropin = proxy_root / "scripts" / "llama-proxy-logrotate"
    assert dropin.exists(), f"Logrotate drop-in missing at {dropin}"
