#!/usr/bin/env python3
"""Tests for the shared proxy-log discovery/opening helper.

`scripts/lib/proxy_logs.py` is the single source of truth for finding and
reading proxy logs across the project scripts. It must handle the deployment's
two rotation schemes (in-process ``proxy.log.*`` and logrotate ``proxy.log-*``),
both plain and gzip-compressed, and never filter by filename-encoded time.

Parent: LP-0MU148SHI004WHQM (child LP-0MUIEK483002TZ54).
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lib import proxy_logs
from lib.proxy_logs import (
    discover_proxy_log_files,
    iter_proxy_log_lines,
    open_proxy_log_text,
)

PLAIN_LINE = "2026-09-26 10:00:00,000 - INFO - live line\n"
DOT_LINE = "2026-09-26 09:00:00,000 - INFO - dot plain line\n"
DASH_LINE = "2026-09-26 08:00:00,000 - INFO - dash plain line\n"
DASH_GZ_LINE = "2026-09-26 07:00:00,000 - INFO - dash gzip line\n"
DOT_GZ_LINE = "2026-09-26 06:00:00,000 - INFO - dot gzip line\n"


def _make_log_dir(tmp_path: Path) -> Path:
    """Create a log dir exercising every naming/compression variant."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text(PLAIN_LINE, encoding="utf-8")
    (log_dir / "proxy.log.2026-09-26_09").write_text(DOT_LINE, encoding="utf-8")
    (log_dir / "proxy.log-2026-09-26_08").write_text(DASH_LINE, encoding="utf-8")
    (log_dir / "proxy.log.2026-09-26_06.gz").write_bytes(
        gzip.compress(DOT_GZ_LINE.encode("utf-8"))
    )
    (log_dir / "proxy.log-2026-09-26_07.gz").write_bytes(
        gzip.compress(DASH_GZ_LINE.encode("utf-8"))
    )
    return log_dir


def test_discover_includes_live_dot_and_dash_plain_and_gz_sorted(tmp_path):
    """AC: discovery covers live + dot + dash, plain + .gz, sorted by name."""
    log_dir = _make_log_dir(tmp_path)

    found = discover_proxy_log_files(log_dir)

    assert [p.name for p in found] == [
        "proxy.log",
        "proxy.log-2026-09-26_07.gz",
        "proxy.log-2026-09-26_08",
        "proxy.log.2026-09-26_06.gz",
        "proxy.log.2026-09-26_09",
    ]
    assert all(p.is_file() for p in found)


def test_discover_ignores_unrelated_files_and_directories(tmp_path):
    """AC: only proxy-log files are returned; matching-name dirs are skipped."""
    log_dir = _make_log_dir(tmp_path)
    (log_dir / "other.log").write_text("nope\n", encoding="utf-8")
    (log_dir / "proxy.log.directory").mkdir()

    names = [p.name for p in discover_proxy_log_files(log_dir)]

    assert "other.log" not in names
    assert "proxy.log.directory" not in names
    assert names[0] == "proxy.log"


def test_discover_returns_empty_for_missing_or_non_directory(tmp_path):
    """AC: a missing path and a regular file both yield an empty list."""
    assert discover_proxy_log_files(tmp_path / "missing") == []
    regular_file = tmp_path / "proxy.log"
    regular_file.write_text(PLAIN_LINE, encoding="utf-8")
    assert discover_proxy_log_files(regular_file) == []


def test_open_proxy_log_text_reads_plain_file(tmp_path):
    """AC: plain files open as UTF-8 text."""
    path = tmp_path / "proxy.log"
    path.write_text(PLAIN_LINE, encoding="utf-8")

    with open_proxy_log_text(path) as fh:
        assert fh.read() == PLAIN_LINE


def test_open_proxy_log_text_decompresses_gz(tmp_path):
    """AC: a .gz file is transparently decompressed, not read as raw bytes."""
    path = tmp_path / "proxy.log-2026-09-26_07.gz"
    path.write_bytes(gzip.compress(DASH_GZ_LINE.encode("utf-8")))

    with open_proxy_log_text(path) as fh:
        assert fh.read() == DASH_GZ_LINE


def test_open_proxy_log_text_replaces_undecodable_bytes(tmp_path):
    """AC: opening never raises on non-UTF-8 bytes (errors='replace')."""
    path = tmp_path / "proxy.log"
    path.write_bytes(b"2026-09-26 10:00:00,000 - INFO - \xff\xfe\n")

    with open_proxy_log_text(path) as fh:
        content = fh.read()

    assert "2026-09-26 10:00:00,000" in content


def test_iter_proxy_log_lines_pairs_every_line_with_its_source(tmp_path):
    """AC: every discovered file's content is yielded as (path, line)."""
    log_dir = _make_log_dir(tmp_path)

    pairs = list(iter_proxy_log_lines(log_dir))

    assert [(p.name, line) for p, line in pairs] == [
        ("proxy.log", PLAIN_LINE),
        ("proxy.log-2026-09-26_07.gz", DASH_GZ_LINE),
        ("proxy.log-2026-09-26_08", DASH_LINE),
        ("proxy.log.2026-09-26_06.gz", DOT_GZ_LINE),
        ("proxy.log.2026-09-26_09", DOT_LINE),
    ]


def test_iter_proxy_log_lines_yields_nothing_for_non_directory(tmp_path):
    """AC: a missing/non-directory path yields no lines."""
    assert list(iter_proxy_log_lines(tmp_path / "missing")) == []


def test_iter_proxy_log_lines_warns_and_continues_on_unreadable_file(
    tmp_path, monkeypatch, capsys
):
    """AC: an OSError on one file warns to stderr and does not drop the rest."""
    log_dir = _make_log_dir(tmp_path)
    unreadable = log_dir / "proxy.log-2026-09-26_08"

    real_open = proxy_logs.open_proxy_log_text

    def fake_open(path):
        if Path(path) == unreadable:
            raise OSError("permission denied")
        return real_open(path)

    monkeypatch.setattr(proxy_logs, "open_proxy_log_text", fake_open)

    pairs = list(iter_proxy_log_lines(log_dir))

    stderr = capsys.readouterr().err
    assert f"warning: cannot read {unreadable}" in stderr
    names = [p.name for p, _ in pairs]
    assert "proxy.log" in names
    assert "proxy.log-2026-09-26_08" not in names
    assert "proxy.log.2026-09-26_09" in names


def test_iter_proxy_log_lines_uses_name_order(tmp_path):
    """AC: iteration order matches discover_proxy_log_files' name ordering."""
    log_dir = _make_log_dir(tmp_path)

    discovered = [p.name for p in discover_proxy_log_files(log_dir)]
    iterated = [p.name for p, _ in iter_proxy_log_lines(log_dir)]

    assert iterated == discovered


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
