#!/usr/bin/env python3
"""Restart-safe proxy logging (LP-0MVLHMAWV008EKTI AC1).

The 2026-10-23 incident: a proxy restart removed a 3 h 35 m window from
``/var/log/llama-proxy``. ``proxy.log``'s first line was the *new* process's
startup line, so the pre-restart interval (18:29:12 -> 22:04:16) was absent
from every retained file, which made the deployed-fix verification window
unmeasurable.

Contract under test (observable behaviour, never internals):

* a non-empty ``proxy.log`` found at startup is **archived** aside (renamed to
  a ``proxy.log.<timestamp>`` file) instead of being truncated or overwritten,
  so a restart can never remove the pre-restart window;
* an existing rotated file is never overwritten by an archive of the same name
  (a name collision must not delete either window);
* an empty ``proxy.log`` produces no archive (restarts without traffic leave
  no junk behind);
* a and real second **process** start preserves the first process's window and
  the shared log-coverage measurement finds no gap across the restart;
* startup archives are pruned/compressed like every other rotated log.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.proxy_logs import measure_log_coverage  # noqa: E402

PRE_RESTART_LINE = "2026-10-23 18:29:12,229 - INFO - pre-restart window line"
ARCHIVE_TS_FORMAT = "%Y-%m-%d_%H%M%S"

#: Each subprocess behaves like one uvicorn start: ``setup_logging`` runs, one
#: line is written, and the handlers are flushed/closed on exit.
_RUNNER = r"""
import json
import logging
import sys

from proxy.server import setup_logging

config = json.loads(sys.argv[1])
logger = setup_logging(config)
logger.info("window marker %s", sys.argv[2])
for handler in logger.handlers:
    handler.flush()
logging.shutdown()
"""


@pytest.fixture
def clean_handlers():
    """Detach any llama-proxy handlers a test (or subprocess helper) left."""
    logger = logging.getLogger("llama-proxy")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    yield
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _config(log_dir: Path, **logging_extra) -> dict:
    logging_cfg = {
        "directory": str(log_dir),
        "level": "INFO",
        "rotation_hours": 6,
        "retention_days": 7,
        "compress_after_days": 1,
    }
    logging_cfg.update(logging_extra)
    return {"logging": logging_cfg}


def _all_log_text(log_dir: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(log_dir.glob("proxy.log*"))
        if path.is_file()
    )


def _run_start(log_dir: Path, marker: str) -> subprocess.CompletedProcess:
    """Run one real process start (the same setup_logging call uvicorn makes)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROXY_DIR)
    env.pop("LLAMA_PROXY_DEV", None)
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, json.dumps(_config(log_dir)), marker],
        cwd=str(PROXY_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


# ---------------------------------------------------------------------------
# AC1 — a startup archives the live log instead of discarding it
# ---------------------------------------------------------------------------


def test_startup_archives_previous_window_instead_of_truncating(
    tmp_path, monkeypatch, clean_handlers
):
    """A non-empty proxy.log is moved aside; its content stays readable."""
    from proxy.server import setup_logging

    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    live = log_dir / "proxy.log"
    live.write_text(PRE_RESTART_LINE + "\n", encoding="utf-8")

    logger = setup_logging(_config(log_dir))
    logger.info("post-restart window line")
    for handler in logger.handlers:
        handler.flush()

    # The live file starts fresh: it no longer carries the pre-restart line.
    assert PRE_RESTART_LINE not in live.read_text(encoding="utf-8")

    archives = [
        path for path in log_dir.glob("proxy.log.*")
        if path.is_file() and PRE_RESTART_LINE in path.read_text(encoding="utf-8")
    ]
    assert len(archives) == 1, (
        "the pre-restart window must be preserved in exactly one archive, "
        f"found: {[p.name for p in log_dir.iterdir()]}"
    )
    # The archive is named for the end of its own content (the live file's
    # last write), so the window stays discoverable without relying on file
    # order or mtime.
    archive_end = dt.datetime.fromtimestamp(archives[0].stat().st_mtime)
    assert archives[0].name.startswith(
        "proxy.log." + archive_end.strftime("%Y-%m-%d_%H")
    ), f"archive name {archives[0].name!r} does not encode its content end"

    # The post-restart line went into the fresh live log, and the restart is
    # attributable from the log itself (the archive move is recorded).
    live_text = live.read_text(encoding="utf-8")
    assert "post-restart window line" in live_text
    assert "Rotated previous proxy log aside" in live_text
    assert archives[0].name in live_text


def test_empty_live_log_is_not_archived(tmp_path, monkeypatch, clean_handlers):
    """A restart with no traffic writes no empty archive."""
    from proxy.server import setup_logging

    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text("", encoding="utf-8")

    setup_logging(_config(log_dir))

    assert list(log_dir.glob("proxy.log.*")) == [], (
        "an empty live log must not produce an archive file"
    )


def test_archive_never_overwrites_an_existing_rotated_file(
    tmp_path, monkeypatch, clean_handlers
):
    """A same-name rotated file is preserved; the archive gets a unique name."""
    from proxy.server import setup_logging

    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    live = log_dir / "proxy.log"
    live.write_text(PRE_RESTART_LINE + "\n", encoding="utf-8")
    # Pin the mtime so the derived archive name is deterministic, but derive
    # it from ``now`` rather than a fixed calendar date: a fixed date becomes
    # a wall-clock time bomb once it ages past ``compress_after_days`` and
    # ``prune_logs`` renames the archive to ``.gz`` before the assertions
    # below (LP-0MVN5PIEG001MJKV).
    pinned = dt.datetime.now().replace(
        minute=0, second=0, microsecond=0
    ).timestamp()
    os.utime(live, (pinned, pinned))
    existing = log_dir / (
        "proxy.log." + dt.datetime.fromtimestamp(pinned).strftime(ARCHIVE_TS_FORMAT)
    )
    existing.write_text("2026-10-23 12:29:16,000 - INFO - older rotated window\n")

    setup_logging(_config(log_dir))

    assert existing.exists(), "an existing rotated file must never be overwritten"
    assert "older rotated window" in existing.read_text(encoding="utf-8")
    archives = [
        path for path in log_dir.glob("proxy.log.*")
        if path != existing and PRE_RESTART_LINE in path.read_text(encoding="utf-8")
    ]
    assert len(archives) == 1, (
        "the pre-restart window must survive in its own archive when the "
        f"derived name is taken, found: {[p.name for p in log_dir.iterdir()]}"
    )


def test_two_process_restart_has_no_coverage_gap(tmp_path):
    """AC1 end-to-end: a real second process start loses no window at all.

    Process 1 writes its window, exits (a restart), process 2 starts. The
    first window must be readable afterwards and the coverage measurement over
    a window spanning the restart must find no gap between the two files.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    first = _run_start(log_dir, "first-process-marker")
    assert first.returncode == 0, first.stderr[-4000:]
    second = _run_start(log_dir, "second-process-marker")
    assert second.returncode == 0, second.stderr[-4000:]

    text = _all_log_text(log_dir)
    assert "first-process-marker" in text, (
        "the pre-restart window was discarded by the second process start"
    )
    assert "second-process-marker" in text

    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert coverage.spans, "no log spans were discovered after the restart"
    assert coverage.gaps == [], (
        "the restart left a coverage gap between the archived window and the "
        f"new live log: {[(g.before.name, g.after.name, g.gap_seconds) for g in coverage.gaps]}"
    )
    assert coverage.covered


def test_startup_archives_are_pruned_like_rotated_logs(
    tmp_path, monkeypatch, clean_handlers
):
    """A startup archive older than retention is deleted; a fresh one is kept."""
    from proxy.server import setup_logging

    monkeypatch.delenv("LLAMA_PROXY_DEV", raising=False)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old = log_dir / (
        "proxy.log."
        + (dt.datetime.now() - dt.timedelta(days=10)).strftime(ARCHIVE_TS_FORMAT)
    )
    old.write_text("2026-10-13 10:00:00,000 - INFO - ancient window\n")

    live = log_dir / "proxy.log"
    live.write_text(PRE_RESTART_LINE + "\n", encoding="utf-8")

    setup_logging(_config(log_dir, retention_days=7, compress_after_days=0))

    assert not old.exists(), "a startup archive past retention must be pruned"
    assert any(
        PRE_RESTART_LINE in path.read_text(encoding="utf-8")
        for path in log_dir.glob("proxy.log.*")
    )
