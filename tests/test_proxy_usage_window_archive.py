"""Tests for ``scripts/proxy_usage_window_archive.py`` (LP-0MVK1PT7Q0005VK6).

The tool is the reproducibility backstop for evidence windows: it copies raw
log segments away before retention rotates them, proves each copy is faithful,
and records the exact command + headline numbers for every regenerated window.

These tests pin the behaviour the evidence depends on:

- **Coverage verdicts.** A window whose segments are entirely rotated away must
  report ``complete: false`` with zero measured hours, and a missing segment
  inside a window must surface as an explicit gap with the omitted hours
  subtracted from ``measured_hours``. Without this, a partial read looks like a
  successful measurement (the failure that motivated the tool).
- **Archive faithfulness.** Every copy is verified (gzip container integrity plus
  byte comparison), the source is left byte-identical, re-runs are idempotent,
  and an archived file whose content diverged from the source is a hard error
  rather than a silent overwrite.
- **Manifest record.** A window block contains the boundaries, the exact command
  the analyst can re-run, the headline numbers parsed from the generated report,
  and the coverage verdict — so AC6 ("every number reproducible from a committed
  command") is falsifiable.
- **llama-server selection.** Those segments carry no per-line timestamps, so
  inclusion must fall back to the name-encoded rotation date; the live
  ``llama-server.log`` is never archived.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "proxy_usage_window_archive.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import proxy_usage_window_archive as archive  # noqa: E402

WINDOW_START = datetime(2026, 10, 20, 14, 44, 27)
WINDOW_END = datetime(2026, 10, 21, 14, 44, 27)


def log_line(ts: str, message: str = "Stream finished: tokens=10/2/12") -> str:
    """One proxy log line in the deployment's ``<ts>,<ms> - LEVEL - …`` shape."""
    return f"{ts},123 - INFO - {message}\n"


def write_segment(path: Path, lines: list[str]) -> Path:
    """Write *lines* to *path*, gzip-compressing ``.gz`` names like the real logs."""
    text = "".join(lines)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def report_md(window: tuple[str, str], sessions: int, requests: int, total: int) -> str:
    """A minimal ``report.md`` carrying the headline fields the tool parses."""
    return (
        "# Proxy usage report\n\n"
        f"- Window: {window[0]} → {window[1]} (24.0h)\n"
        f"- Sessions in window: **{sessions}** | Requests: **{requests}** "
        f"(local 561 / remote {requests - 561})\n\n"
        "## Total tokens\n\n"
        "| Kind | Count | % of total |\n"
        "|---|---|---|\n"
        "| Input (prompt) | 756,627,403 | 99.7% |\n"
        "| Output (completion) | 2,620,664 | 0.3% |\n"
        f"| Total | **{total:,}** | 100.0% |\n"
    )


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Two adjacent proxy segments covering the front half of the window."""
    directory = tmp_path / "logs"
    directory.mkdir()
    write_segment(
        directory / "proxy.log.2026-10-20_15.gz",
        [log_line("2026-10-20 14:44:27"), log_line("2026-10-20 21:49:56")],
    )
    write_segment(
        directory / "proxy.log.2026-10-20_21.gz",
        [log_line("2026-10-21 00:00:02"), log_line("2026-10-21 08:00:00")],
    )
    return directory


class TestCoverage:
    def test_reports_measured_hours_and_missing_tail(self, log_dir: Path) -> None:
        payload = archive.window_coverage(log_dir, WINDOW_START, WINDOW_END)

        assert payload["window_hours"] == 24.0
        assert payload["complete"] is False
        # 14:44:27 → 21:49:56 and 00:00:02 → 08:00:00 plus the 2.17 h hole between
        # the two segments: 7.09 + 8.00 h of content, 2.17 h gap.
        assert payload["measured_hours"] == pytest.approx(15.09, abs=0.05)
        assert len(payload["gaps"]) == 1
        assert payload["tail_missing_hours"] > 6.0
        assert [record["name"] for record in payload["files"]] == [
            "proxy.log.2026-10-20_15.gz",
            "proxy.log.2026-10-20_21.gz",
        ]

    def test_missing_segment_inside_window_is_reported_as_gap(self, tmp_path: Path) -> None:
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(
            directory / "proxy.log.2026-10-20_15.gz",
            [log_line("2026-10-20 14:44:27"), log_line("2026-10-20 21:49:56")],
        )
        # The 2026-10-21 00:00 daily rotation is missing (the real 2.17 h hole).
        write_segment(
            directory / "proxy.log.2026-10-21_07.gz",
            [log_line("2026-10-21 03:50:01"), log_line("2026-10-21 14:44:27")],
        )

        payload = archive.window_coverage(directory, WINDOW_START, WINDOW_END)

        assert payload["complete"] is False
        assert len(payload["gaps"]) == 1
        gap = payload["gaps"][0]
        assert gap["before"] == "proxy.log.2026-10-20_15.gz"
        assert gap["after"] == "proxy.log.2026-10-21_07.gz"
        assert gap["gap_hours"] == pytest.approx(6.0, abs=0.05)
        # 24 h window with a 6 h hole in the middle.
        assert payload["measured_hours"] == pytest.approx(18.0, abs=0.1)

    def test_fully_rotated_window_reports_zero_coverage(self, tmp_path: Path) -> None:
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(
            directory / "proxy.log.2026-10-20_15.gz",
            [log_line("2026-10-20 20:00:00"), log_line("2026-10-20 21:00:00")],
        )

        payload = archive.window_coverage(
            directory, datetime(2026, 10, 18, 14, 37, 48), datetime(2026, 10, 19, 14, 37, 48)
        )

        assert payload["files"] == []
        assert payload["measured_hours"] == 0.0
        assert payload["covered_fraction"] == 0.0
        assert payload["complete"] is False

    def test_missing_log_dir_is_not_an_error(self, tmp_path: Path) -> None:
        payload = archive.window_coverage(tmp_path / "absent", WINDOW_START, WINDOW_END)

        assert payload["complete"] is False
        assert payload["measured_hours"] == 0.0

    def test_contiguous_segments_report_a_complete_window(self, tmp_path: Path) -> None:
        """A window whose retained lines span it with no hole is complete.

        The last line is a few seconds before ``window_end`` (as in production,
        where a window boundary is chosen from a report header): that sub-minute
        slack must not be mistaken for a truncated measurement.
        """
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(
            directory / "proxy.log.2026-10-20_15.gz",
            [log_line("2026-10-20 14:44:27"), log_line("2026-10-21 00:00:00")],
        )
        write_segment(
            directory / "proxy.log.2026-10-21_07.gz",
            [log_line("2026-10-21 00:00:04"), log_line("2026-10-21 14:44:25")],
        )

        payload = archive.window_coverage(directory, WINDOW_START, WINDOW_END)

        assert payload["gaps"] == []
        assert payload["measured_hours"] == pytest.approx(24.0, abs=0.01)
        assert payload["complete"] is True


class TestReportParsing:
    def test_extracts_headline_numbers_from_report(self, tmp_path: Path) -> None:
        path = tmp_path / "report.md"
        path.write_text(
            report_md(("2026-10-20 14:44:27", "2026-10-21 14:44:27"), 214, 8359, 759_248_067),
            encoding="utf-8",
        )

        headline = archive.parse_report_headline(path)

        assert headline["sessions"] == 214
        assert headline["requests"] == 8359
        assert headline["local_requests"] == 561
        assert headline["remote_requests"] == 7798
        assert headline["input_tokens"] == 756_627_403
        assert headline["output_tokens"] == 2_620_664
        assert headline["total_tokens"] == 759_248_067
        assert headline["tokens_per_request"] == pytest.approx(90830, abs=1)

    def test_absent_report_yields_none_fields(self, tmp_path: Path) -> None:
        headline = archive.parse_report_headline(tmp_path / "missing.md")

        assert headline["sessions"] is None
        assert headline["total_tokens"] is None
        assert headline["tokens_per_request"] is None


class TestArchive:
    def test_copies_verifies_and_leaves_source_untouched(self, log_dir: Path, tmp_path: Path) -> None:
        dest = tmp_path / "raw"
        sources = archive.files_in_range(log_dir, WINDOW_START, WINDOW_END)
        originals = {path.name: path.read_bytes() for path in sources}

        records = archive.archive_files(sources, dest)

        assert len(records) == 2
        for record in records:
            assert record.verification == "gzip+cmp"
            assert record.copied is True
            assert (dest / record.name).read_bytes() == originals[record.name]
            assert record.sha256 == archive._sha256(dest / record.name)
        # Sources are read-only inputs.
        for name, content in originals.items():
            assert (log_dir / name).read_bytes() == content

    def test_rerun_is_idempotent(self, log_dir: Path, tmp_path: Path) -> None:
        dest = tmp_path / "raw"
        sources = archive.files_in_range(log_dir, WINDOW_START, WINDOW_END)

        archive.archive_files(sources, dest)
        second = archive.archive_files(sources, dest)

        assert [record.copied for record in second] == [False, False]
        assert [record.verification for record in second] == ["gzip+cmp", "gzip+cmp"]

    def test_refuses_to_overwrite_diverged_archive(self, log_dir: Path, tmp_path: Path) -> None:
        dest = tmp_path / "raw"
        dest.mkdir()
        sources = archive.files_in_range(log_dir, WINDOW_START, WINDOW_END)
        # A pre-existing archive copy with different content must not be replaced.
        write_segment(dest / sources[0].name, [log_line("2026-10-20 14:44:27", "different")])

        with pytest.raises(ValueError, match="refusing to overwrite"):
            archive.archive_files(sources, dest)

    def test_corrupt_existing_copy_is_rejected(self, log_dir: Path, tmp_path: Path) -> None:
        dest = tmp_path / "raw"
        dest.mkdir()
        sources = archive.files_in_range(log_dir, WINDOW_START, WINDOW_END)
        # Same bytes, but not a valid gzip stream: the copy is not faithful.
        corrupt = dest / sources[0].name
        corrupt.write_bytes(b"not a gzip stream\n")

        with pytest.raises(ValueError):
            archive.archive_files([sources[0]], dest)

    def test_checksums_cover_every_archived_file(self, log_dir: Path, tmp_path: Path) -> None:
        dest = tmp_path / "raw"
        sources = archive.files_in_range(log_dir, WINDOW_START, WINDOW_END)
        archive.archive_files(sources, dest)
        # Derived records next to the segments must not invalidate the set.
        (dest / "manifest.txt").write_text("# manifest\n", encoding="utf-8")
        (dest / "archive.json").write_text("{}", encoding="utf-8")

        sums = archive.write_checksums(dest)

        lines = sums.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(".gz" in line for line in lines)
        assert all(len(line.split()[0]) == 64 for line in lines)
        names = [line.split()[-1] for line in lines]
        assert names == sorted(names)  # deterministic (by filename) ordering
        # Idempotent: re-running yields byte-identical checksums.
        assert archive.write_checksums(dest).read_text(encoding="utf-8") == sums.read_text(encoding="utf-8")

    def test_selection_uses_content_not_filename(self, tmp_path: Path) -> None:
        """A rotated file routinely holds data past its name-encoded hour."""
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(
            directory / "proxy.log.2026-10-19_03.gz",
            [log_line("2026-10-19 02:00:00"), log_line("2026-10-19 09:03:00")],
        )
        write_segment(
            directory / "proxy.log.2026-10-25_03.gz",
            [log_line("2026-10-25 02:00:00"), log_line("2026-10-25 09:03:00")],
        )

        selected = [
            path.name
            for path in archive.files_in_range(
                directory, datetime(2026, 10, 19, 0, 0, 0), datetime(2026, 10, 19, 23, 59, 59)
            )
        ]

        assert selected == ["proxy.log.2026-10-19_03.gz"]

    def test_llama_server_segments_selected_by_rotation_date(self, tmp_path: Path) -> None:
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(directory / "llama-server.log-2026-10-20.gz", ["srv  load: spawning\n"])
        write_segment(directory / "llama-server.log-2026-10-25.gz", ["srv  load: spawning\n"])
        (directory / "llama-server.log").write_text("srv  live\n", encoding="utf-8")
        write_segment(directory / "proxy.log.2026-10-20_15.gz", [log_line("2026-10-20 15:00:00")])

        selected = sorted(path.name for path in archive.files_in_range(directory, WINDOW_START, WINDOW_END))

        # The 10-25 segment and the live log fall outside the window.
        assert selected == ["llama-server.log-2026-10-20.gz", "proxy.log.2026-10-20_15.gz"]

    def test_cli_archive_reports_json(self, log_dir: Path, tmp_path: Path, capsys) -> None:
        dest = tmp_path / "raw"

        exit_code = archive.main(
            [
                "archive",
                "--log-dir",
                str(log_dir),
                "--dest",
                str(dest),
                "--start",
                "2026-10-20 14:44:27",
                "--end",
                "2026-10-21 14:44:27",
                "--json",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 2
        assert payload["copied"] == 2
        assert payload["coverage"]["complete"] is False
        assert Path(payload["checksums"]).is_file()

    def test_cli_archive_fails_when_nothing_is_retained(self, tmp_path: Path, capsys) -> None:
        directory = tmp_path / "logs"
        directory.mkdir()
        write_segment(directory / "proxy.log.2026-10-25_03.gz", [log_line("2026-10-25 02:00:00")])

        exit_code = archive.main(
            [
                "archive",
                "--log-dir",
                str(directory),
                "--dest",
                str(tmp_path / "raw"),
                "--start",
                "2026-10-18 00:00:00",
                "--end",
                "2026-10-18 23:59:59",
            ]
        )

        assert exit_code == 1
        assert "nothing to archive" in capsys.readouterr().err


class TestRunWindow:
    def _stub_analyzer(self, tmp_path: Path, *, exit_code: int = 0) -> Path:
        """A fake analysis CLI that records its argv and writes a report."""
        stub = tmp_path / "stub_analyzer.py"
        stub.write_text(
            "import json, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'argv.json').write_text(json.dumps(args))\n"
            "(out / 'report.md').write_text("
            f"{report_md(('2026-10-20 14:44:27', '2026-10-21 14:44:27'), 214, 8359, 759_248_067)!r})\n"
            "print('stub analysis done')\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        return stub

    def test_records_command_and_headline_in_manifest(self, log_dir: Path, tmp_path: Path) -> None:
        stub = self._stub_analyzer(tmp_path)
        manifest = tmp_path / "raw" / "manifest.txt"

        result = archive.run_window(
            name="post24",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "post24",
            log_dir=log_dir,
            analyzer=stub,
            manifest_path=manifest,
        )

        assert result["returncode"] == 0
        assert result["headline"]["total_tokens"] == 759_248_067
        recorded_argv = json.loads((tmp_path / "post24" / "argv.json").read_text(encoding="utf-8"))
        assert recorded_argv[:6] == [
            "--log-dir",
            str(log_dir),
            "--start",
            "2026-10-20 14:44:27",
            "--end",
            "2026-10-21 14:44:27",
        ]
        # The manifest command is the one that actually ran.
        text = manifest.read_text(encoding="utf-8")
        assert "### post24" in text
        assert str(stub) in text
        assert '--start "2026-10-20 14:44:27"' in text
        assert "Requests: 8359" in text
        assert "759,248,067" in text
        assert "INCOMPLETE" in text  # the fixture's window has a missing tail
        assert (tmp_path / "post24" / "cli-stdout.txt").read_text(encoding="utf-8").startswith("stub analysis done")

    def test_failed_run_writes_no_manifest_block(self, log_dir: Path, tmp_path: Path) -> None:
        stub = self._stub_analyzer(tmp_path, exit_code=3)
        manifest = tmp_path / "manifest.txt"

        result = archive.run_window(
            name="broken",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "broken",
            log_dir=log_dir,
            analyzer=stub,
            manifest_path=manifest,
        )

        assert result["returncode"] == 3
        assert "manifest" not in result
        assert not manifest.exists()

    def test_manifest_header_created_once_and_appended_per_window(self, log_dir: Path, tmp_path: Path) -> None:
        stub = self._stub_analyzer(tmp_path)
        manifest = tmp_path / "manifest.txt"

        for name in ("post24", "now24"):
            archive.run_window(
                name=name,
                window_start=WINDOW_START,
                window_end=WINDOW_END,
                output_dir=tmp_path / name,
                log_dir=log_dir,
                analyzer=stub,
                manifest_path=manifest,
            )

        text = manifest.read_text(encoding="utf-8")
        assert text.count("# Proxy-usage window manifest") == 1
        assert text.count("### post24") == 1
        assert text.count("### now24") == 1
        assert text.index("### post24") < text.index("### now24")

    def test_cli_run_window_exits_nonzero_on_analyzer_failure(self, log_dir: Path, tmp_path: Path) -> None:
        stub = self._stub_analyzer(tmp_path, exit_code=2)

        exit_code = archive.main(
            [
                "run-window",
                "--name",
                "bad",
                "--start",
                "2026-10-20 14:44:27",
                "--end",
                "2026-10-21 14:44:27",
                "--output-dir",
                str(tmp_path / "bad"),
                "--log-dir",
                str(log_dir),
                "--analyzer",
                str(stub),
            ]
        )

        assert exit_code == 1


class TestManifestRendering:
    def test_lists_gaps_and_missing_lead_in(self) -> None:
        coverage = {
            "window_start": "2026-10-19 14:44:27",
            "window_end": "2026-10-20 14:44:27",
            "window_hours": 24.0,
            "measured_hours": 1.57,
            "covered_fraction": 0.0654,
            "complete": False,
            "lead_in_missing_hours": 22.43,
            "tail_missing_hours": 0.0,
            "files": [],
            "gaps": [],
        }
        block = archive.render_manifest_block(
            "ref24",
            "python3 analyze.py",
            Path("/tmp/ref24"),
            {},
            coverage,
            generated_at=datetime(2026, 10, 28, 19, 0, 0),
        )

        assert "missing lead-in: 22.43 h" in block
        assert "INCOMPLETE" in block
        assert "Generated: 2026-10-28 19:00:00" in block

    def test_shell_command_quotes_spaces(self) -> None:
        command = archive.shell_command(["prog", "--start", "2026-10-20 14:44:27", "--json"])

        assert command == 'prog --start "2026-10-20 14:44:27" --json'
