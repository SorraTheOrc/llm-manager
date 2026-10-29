#!/usr/bin/env python3
"""Archive raw proxy log segments and regenerate analysis windows reproducibly.

Why this exists (LP-0MVK1PT7Q0005VK6)
-------------------------------------

Rotated ``proxy.log`` segments are not retained contractually: in this
deployment they last about a week. An investigation that quotes numbers from a
24 h window therefore has to (a) copy the raw segments that cover the window
somewhere safe **before** retention runs, and (b) record the exact command that
produced every number, so the figures stay reproducible after rotation.

An earlier pass measured four windows (2026-10-18 .. 2026-10-22) into ``/tmp``
and lost the outputs; by the time this tool was written the 2026-10-18/19
segments had already rotated away. That is the failure mode this tool makes
visible and, for future windows, avoidable.

What it does
------------

``coverage``
    Report, per log file, the first/last timestamp of the lines it contributes
    to a window, plus any adjacent-file gap inside the window. Discovery,
    gzip handling and gap detection are the shared, already-tested
    :mod:`lib.proxy_logs` implementation (one source of truth for the two
    rotation naming schemes), so this subcommand is a thin CLI over it.

``archive``
    Copy every log file in the log dir that contributes lines to a window into
    a destination directory, verify each copy (``gzip -t`` for compressed
    segments, byte comparison otherwise), record SHA-256 sums, and refuse to
    overwrite an archived file whose content differs from the source. The
    source log directory is never written to.

``run-window``
    Run the proxy-usage-analysis CLI for one named window, capture its stdout,
    parse the headline numbers out of the generated ``report.md``, and append a
    manifest block (window boundaries, exact command, headline numbers,
    measured coverage) to the manifest file. The manifest is the record that
    parent AC6 ("every number quoted is reproducible from a committed
    command") depends on.

All subcommands support ``--json`` for machine-readable output (project rule:
CLI output is available as JSON).

Usage
-----

    python3 scripts/proxy_usage_window_archive.py coverage \\
        --log-dir /var/log/llama-proxy \\
        --start "2026-10-20 14:44:27" --end "2026-10-21 14:44:27" --json

    python3 scripts/proxy_usage_window_archive.py archive \\
        --log-dir /var/log/llama-proxy \\
        --dest ~/proxy-usage-reports/cost/raw \\
        --start "2026-10-18 00:00:00" --end "2026-10-22 23:59:59"

    python3 scripts/proxy_usage_window_archive.py run-window \\
        --name post24 --start "2026-10-20 14:44:27" --end "2026-10-21 14:44:27" \\
        --output-dir ~/proxy-usage-reports/cost/post24 \\
        --manifest ~/proxy-usage-reports/cost/raw/manifest.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.proxy_logs import (  # noqa: E402
    LogCoverage,
    discover_proxy_log_files,
    measure_log_coverage,
    open_proxy_log_text,
    parse_line_timestamp,
)

#: The analysis CLI this tool drives (used as-is; never modified here).
DEFAULT_ANALYZER = _REPO_ROOT / ".pi" / "skills" / "proxy-usage-analysis" / "scripts" / "analyze_proxy_usage.py"

#: Second log stream archived alongside ``proxy.log`` (decode/prompt-eval stats).
DEFAULT_LLAMA_PREFIX = "llama-server.log"

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

#: A window counts as fully covered when no segment gap exists and the retained
#: lines span the window to within this slack. Log timestamps have one-second
#: granularity and windows are chosen from a report's own boundaries, so the
#: last retained line routinely lands a second or two before ``window_end`` —
#: a sub-minute slack keeps that from reading as a truncated measurement.
_COMPLETENESS_TOLERANCE_SECONDS = 60.0

#: ``- Sessions in window: **214** | Requests: **8359** (local 561 / remote 7798)``
_SESSIONS_RE = re.compile(
    r"Sessions in window:\s*\*\*(?P<sessions>\d+)\*\*\s*\|\s*Requests:\s*\*\*(?P<requests>\d+)\*\*"
    r"\s*\(local\s+(?P<local>\d+)\s*/\s*remote\s+(?P<remote>\d+)\)"
)
#: ``- Window: 2026-10-20 14:44:27 → 2026-10-21 14:44:27 (24.0h)``
_WINDOW_RE = re.compile(r"^- Window:\s*(?P<start>.+?) → (?P<end>.+?) \(")
#: ``| Total | **759,248,067** | 100.0% |`` (only the Total row is bolded)
_TOKEN_ROW_RE = re.compile(
    r"^\|\s*(?P<kind>Input \(prompt\)|Output \(completion\)|Total)\s*\|\s*\**\s*(?P<value>[\d,]+)"
)


def parse_ts(value: str) -> datetime:
    """Parse the CLI's ``YYYY-MM-DD HH:MM:SS`` window boundary format."""
    try:
        return datetime.strptime(value.strip(), _TS_FORMAT)
    except ValueError:
        return datetime.fromisoformat(value.strip())


def fmt_ts(value: datetime | None) -> str:
    return value.strftime(_TS_FORMAT) if value else "-"


def parse_report_headline(report_path: Path) -> dict:
    """Extract the headline numbers from a generated ``report.md``.

    Reads only what the report itself states (sessions/requests per bucket and
    the ``## Total tokens`` table), so a manifest can never disagree with the
    artifact it describes. Missing fields degrade to ``None`` rather than
    raising, so an older/short report still yields a usable partial record.
    """
    headline: dict[str, object] = {
        "window_start": None,
        "window_end": None,
        "sessions": None,
        "requests": None,
        "local_requests": None,
        "remote_requests": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "tokens_per_request": None,
    }
    if not report_path.is_file():
        return headline
    in_tokens = False
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_tokens = stripped == "## Total tokens"
        if match := _WINDOW_RE.match(stripped):
            headline["window_start"] = match.group("start").strip()
            headline["window_end"] = match.group("end").strip()
        if match := _SESSIONS_RE.search(stripped):
            headline["sessions"] = int(match.group("sessions"))
            headline["requests"] = int(match.group("requests"))
            headline["local_requests"] = int(match.group("local"))
            headline["remote_requests"] = int(match.group("remote"))
        if in_tokens and (match := _TOKEN_ROW_RE.match(stripped)):
            value = int(match.group("value").replace(",", ""))
            kind = match.group("kind")
            if kind.startswith("Input"):
                headline["input_tokens"] = value
            elif kind.startswith("Output"):
                headline["output_tokens"] = value
            else:
                headline["total_tokens"] = value
    requests = headline["requests"]
    total = headline["total_tokens"]
    if isinstance(requests, int) and requests > 0 and isinstance(total, int):
        headline["tokens_per_request"] = int(round(total / requests))
    return headline


def coverage_payload(coverage: LogCoverage, window_start: datetime, window_end: datetime) -> dict:
    """Serialise a :class:`LogCoverage` into the manifest/JSON shape.

    ``window_seconds`` is the requested window; ``measured_seconds`` is the part
    of it that at least one retained file actually covers, computed from the
    spans and the in-window gaps. The two together are what let a reader tell a
    genuine measurement from a silently partial one.
    """
    window_seconds = (window_end - window_start).total_seconds()
    spans = sorted(coverage.spans, key=lambda s: s.first)
    measured = 0.0
    if spans:
        first_covered = max(spans[0].first, window_start)
        last_covered = min(max(span.last for span in spans), window_end)
        measured = max(0.0, (last_covered - first_covered).total_seconds())
        measured -= sum(g.gap_seconds for g in coverage.gaps)
        measured = max(0.0, measured)
    return {
        "window_start": fmt_ts(window_start),
        "window_end": fmt_ts(window_end),
        "window_hours": round(window_seconds / 3600, 2),
        "measured_hours": round(measured / 3600, 2),
        "covered_fraction": round(measured / window_seconds, 4) if window_seconds else 0.0,
        "complete": bool(spans) and not coverage.gaps and measured >= window_seconds - _COMPLETENESS_TOLERANCE_SECONDS,
        "lead_in_missing_hours": (
            round(coverage.window_start_gap_seconds / 3600, 2) if coverage.window_start_gap_seconds else 0.0
        ),
        "tail_missing_hours": (
            round(coverage.window_end_gap_seconds / 3600, 2) if coverage.window_end_gap_seconds else 0.0
        ),
        "files": [
            {
                "name": span.path.name,
                "first": fmt_ts(span.first),
                "last": fmt_ts(span.last),
                "lines": span.line_count,
            }
            for span in coverage.spans
        ],
        "gaps": [
            {
                "before": gap.before.name,
                "before_last": fmt_ts(gap.before_last),
                "after": gap.after.name,
                "after_first": fmt_ts(gap.after_first),
                "gap_hours": round(gap.gap_seconds / 3600, 2),
            }
            for gap in coverage.gaps
        ],
    }


def window_coverage(
    log_dir: Path, window_start: datetime, window_end: datetime, max_gap_seconds: float = 3600.0
) -> dict:
    """Measure how completely *log_dir* covers ``[window_start, window_end]``."""
    return coverage_payload(
        measure_log_coverage(log_dir, window_start, window_end, max_gap_seconds),
        window_start,
        window_end,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_copy(source: Path, copy: Path) -> str:
    """Return ``"gzip"``/``"cmp"`` when *copy* is a faithful copy of *source*.

    Compressed segments are checked for container integrity **and** compared
    byte-for-byte; plain segments are compared byte-for-byte. The byte
    comparison is what makes this a *copy* verification rather than only a
    "the gzip stream parses" check.
    """
    if source.name.endswith(".gz"):
        try:
            with open_proxy_log_text(copy) as handle:
                for _ in handle:
                    pass
        except OSError as exc:  # includes gzip.BadGzipFile / EOFError
            raise ValueError(f"compressed copy {copy.name} failed integrity check: {exc}") from exc
        if copy.read_bytes() != source.read_bytes():
            raise ValueError(f"copy {copy.name} differs from source {source}")
        return "gzip+cmp"
    if copy.read_bytes() != source.read_bytes():
        raise ValueError(f"copy {copy.name} differs from source {source}")
    return "cmp"


#: Daily-rotated llama-server segment name: ``llama-server.log-2026-10-20.gz``
_LLAMA_DATE_RE = re.compile(rf"{re.escape(DEFAULT_LLAMA_PREFIX)}-(\d{{4}})-(\d{{2}})-(\d{{2}})(?:\.gz)?$")


def _llama_name_date(name: str) -> datetime | None:
    """Return the rotation date encoded in a llama-server segment name.

    llama-server logs carry **no** per-line timestamps (verified in this
    deployment), so the name-encoded rotation date is the only boundary signal
    they expose; inclusion decisions must use it.
    """
    match = _LLAMA_DATE_RE.match(name)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return datetime(year, month, day)


def files_in_range(
    log_dir: Path,
    window_start: datetime,
    window_end: datetime,
    llama_log_dir: Path | None = None,
) -> list[Path]:
    """Return the log files whose content intersects ``[window_start, window_end]``.

    Proxy files come from the shared discovery helper (both rotation schemes,
    plain or gzip); their contribution is decided by the actual first/last
    timestamps of retained lines. ``llama-server.log*`` segments have no line
    timestamps, so their daily rotation date (from the filename) is used as the
    boundary instead; the live ``llama-server.log`` is never archived (it
    carries only the current day and cannot be ranged).
    """
    selected: list[Path] = []
    for path in discover_proxy_log_files(log_dir):
        first, last = _line_span(path)
        if first is not None and last is not None and first <= window_end and last >= window_start:
            selected.append(path)
    other_dir = Path(llama_log_dir) if llama_log_dir else Path(log_dir)
    if other_dir.is_dir():
        for path in sorted(other_dir.iterdir()):
            if not (path.is_file() and path.name.startswith(DEFAULT_LLAMA_PREFIX)):
                continue
            encoded = _llama_name_date(path.name)
            if encoded is None:
                continue  # live/unranged llama-server.log
            if window_start.date() <= encoded.date() <= window_end.date():
                selected.append(path)
    return selected


def _line_span(path: Path) -> tuple[datetime | None, datetime | None]:
    first: datetime | None = None
    last: datetime | None = None
    try:
        with open_proxy_log_text(path) as handle:
            for line in handle:
                ts = parse_line_timestamp(line)
                if ts is None:
                    continue
                if first is None or ts < first:
                    first = ts
                if last is None or ts > last:
                    last = ts
    except OSError:
        return None, None
    return first, last


@dataclass(frozen=True)
class ArchiveRecord:
    """One archived segment: source, copy, integrity verification, digest."""

    name: str
    source: str
    dest: str
    bytes: int
    sha256: str
    verification: str
    copied: bool

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "dest": self.dest,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "verification": self.verification,
            "copied": self.copied,
        }


def archive_files(sources: list[Path], dest_dir: Path) -> list[ArchiveRecord]:
    """Copy *sources* into *dest_dir*, verifying every copy.

    Existing archives are never overwritten: an identical copy is left alone
    (idempotent re-runs) and a differing copy is a hard error, because a
    diverged archive means the evidence changed underneath us. Source files are
    only ever read.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    records: list[ArchiveRecord] = []
    for source in sources:
        dest = dest_dir / source.name
        source_digest = _sha256(source)
        copied = False
        if dest.exists():
            existing = _sha256(dest)
            if existing != source_digest:
                raise ValueError(
                    f"refusing to overwrite {dest}: archived content differs from {source} "
                    f"(archive sha256 {existing[:12]}, source sha256 {source_digest[:12]})"
                )
        else:
            shutil.copy2(source, dest)
            copied = True
        verification = _verify_copy(source, dest)
        records.append(
            ArchiveRecord(
                name=source.name,
                source=str(source),
                dest=str(dest),
                bytes=dest.stat().st_size,
                sha256=source_digest,
                verification=verification,
                copied=copied,
            )
        )
    return records


def _is_log_segment(name: str) -> bool:
    """True for archived log segments (proxy and llama-server), not derived records.

    Derived files written *next to* the segments (``manifest.txt``,
    ``archive.json``, ``README.md``, this checksum file) are deliberately
    excluded: hashing the record while it is being produced would make the
    checksum set self-referential and unstable.
    """
    return (
        name == "proxy.log"
        or name.startswith("proxy.log.")
        or name.startswith("proxy.log-")
        or name.startswith(DEFAULT_LLAMA_PREFIX)
    )


def write_checksums(dest_dir: Path) -> Path:
    """Write ``SHA256SUMS`` for every archived log segment in *dest_dir*.

    Segments are re-hashed (not taken from the copy records) so the checksum
    set always covers the archive as it actually is — including segments copied
    outside this tool's selection and re-runs that skip existing copies.
    """
    path = Path(dest_dir) / "SHA256SUMS"
    dest_dir = Path(dest_dir)
    lines = []
    for candidate in sorted(p for p in dest_dir.iterdir() if p.is_file() and _is_log_segment(p.name)):
        lines.append(f"{_sha256(candidate)}  {candidate.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def shell_command(argv: list[str]) -> str:
    """Render *argv* as a copy-pasteable shell command (quoted where needed)."""
    return " ".join(shlex_quote(token) for token in argv)


def shlex_quote(token: str) -> str:
    if token and all(char.isalnum() or char in "-_=/:.,+@%^" for char in token):
        return token
    escaped = token.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_manifest_block(
    name: str,
    command: str,
    output_dir: Path,
    headline: dict,
    coverage: dict,
    generated_at: datetime | None = None,
) -> str:
    """Render one window's manifest block (markdown, appended per run)."""
    stamp = (generated_at or datetime.now()).strftime(_TS_FORMAT)
    lines = [f"### {name}", ""]
    lines.append(
        f"- Window: {headline.get('window_start') or coverage['window_start']} → "
        f"{headline.get('window_end') or coverage['window_end']}"
    )
    lines.append(f"- Output dir: {output_dir}")
    lines.append(f"- Command: `{command}`")
    if headline.get("sessions") is not None:
        lines.append(
            f"- Sessions: {headline['sessions']} | Requests: {headline['requests']} "
            f"(local {headline['local_requests']} / remote {headline['remote_requests']})"
        )
    if headline.get("total_tokens") is not None:
        lines.append(
            f"- Logged tokens (input + output): {headline['input_tokens']:,} + "
            f"{headline['output_tokens']:,} = **{headline['total_tokens']:,}**"
        )
        lines.append(f"- Tokens per request: {headline['tokens_per_request']:,}")
    coverage_line = (
        f"- Log coverage: {coverage['measured_hours']} h of {coverage['window_hours']} h "
        f"({coverage['covered_fraction'] * 100:.1f}%) across {len(coverage['files'])} file(s)"
    )
    lines.append(coverage_line + (" — complete" if coverage["complete"] else " — INCOMPLETE"))
    for gap in coverage["gaps"]:
        lines.append(
            f"  - gap {gap['gap_hours']} h: {gap['before']} ends {gap['before_last']} → "
            f"{gap['after']} starts {gap['after_first']}"
        )
    if coverage["lead_in_missing_hours"]:
        lines.append(f"  - missing lead-in: {coverage['lead_in_missing_hours']} h before the first retained line")
    if coverage["tail_missing_hours"]:
        lines.append(f"  - missing tail: {coverage['tail_missing_hours']} h after the last retained line")
    lines.append(f"- Generated: {stamp}")
    lines.append("")
    return "\n".join(lines)


def append_manifest(manifest_path: Path, block: str) -> None:
    """Append *block* to the manifest, creating it with a header when absent."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(
            "# Proxy-usage window manifest\n\n"
            "One block per regenerated window: boundaries, the exact command, the headline\n"
            "numbers printed by the analysis CLI, and the log coverage actually measured.\n"
            "Produced by `scripts/proxy_usage_window_archive.py run-window`.\n\n",
            encoding="utf-8",
        )
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def run_window(
    name: str,
    window_start: datetime,
    window_end: datetime,
    output_dir: Path,
    log_dir: Path,
    analyzer: Path = DEFAULT_ANALYZER,
    llama_log_dir: Path | None = None,
    manifest_path: Path | None = None,
    extra_args: list[str] | None = None,
    python_executable: str | None = None,
) -> dict:
    """Run the analysis CLI for one window and optionally record it in the manifest.

    The command is built (and recorded) verbatim, so the manifest line can be
    re-executed to reproduce the numbers. Non-zero CLI exit codes are reported
    in the result rather than raised, but no manifest block is written for a
    failed run — the manifest only ever describes an artifact that exists.
    """
    output_dir = Path(output_dir).expanduser()
    argv = [
        python_executable or sys.executable,
        str(analyzer),
        "--log-dir",
        str(log_dir),
        "--start",
        fmt_ts(window_start),
        "--end",
        fmt_ts(window_end),
        "--output-dir",
        str(output_dir),
    ]
    if llama_log_dir is not None:
        argv += ["--llama-log-dir", str(llama_log_dir)]
    argv += list(extra_args or [])
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cli-stdout.txt").write_text(
        completed.stdout + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    headline = parse_report_headline(output_dir / "report.md")
    coverage = window_coverage(log_dir, window_start, window_end)
    result = {
        "name": name,
        "returncode": completed.returncode,
        "command": shell_command(argv),
        "output_dir": str(output_dir),
        "headline": headline,
        "coverage": coverage,
    }
    if completed.returncode == 0 and manifest_path is not None and (output_dir / "report.md").is_file():
        append_manifest(
            Path(manifest_path),
            render_manifest_block(name, result["command"], output_dir, headline, coverage),
        )
        result["manifest"] = str(manifest_path)
    return result


def _print_window_table(payload: dict) -> None:
    print(f"Window: {payload['window_start']} → {payload['window_end']} ({payload['window_hours']} h)")
    print(
        f"Coverage: {payload['measured_hours']} h measured "
        f"({payload['covered_fraction'] * 100:.1f}%) — "
        f"{'complete' if payload['complete'] else 'INCOMPLETE'}"
    )
    if payload["files"]:
        print(f"{'file':<44} {'first':<20} {'last':<20} {'lines':>8}")
        for record in payload["files"]:
            print(f"{record['name']:<44} {record['first']:<20} {record['last']:<20} {record['lines']:>8}")
    else:
        print("No retained log file contributes lines to this window.")
    for gap in payload["gaps"]:
        print(
            f"gap: {gap['before']} ends {gap['before_last']} → "
            f"{gap['after']} starts {gap['after_first']} ({gap['gap_hours']} h)"
        )


def _cmd_coverage(args: argparse.Namespace) -> int:
    payload = window_coverage(
        Path(args.log_dir).expanduser(), parse_ts(args.start), parse_ts(args.end), args.max_gap_seconds
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_window_table(payload)
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    start, end = parse_ts(args.start), parse_ts(args.end)
    llama_dir = Path(args.llama_log_dir).expanduser() if args.llama_log_dir else None
    sources = files_in_range(Path(args.log_dir).expanduser(), start, end, llama_dir)
    if not sources and not args.allow_empty:
        print(
            "error: no log file in the range has retained lines; nothing to archive "
            "(pass --allow-empty to record the empty archive anyway)",
            file=sys.stderr,
        )
        return 1
    try:
        records = archive_files(sources, Path(args.dest).expanduser())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    checksums = write_checksums(Path(args.dest).expanduser())
    payload = {
        "dest": str(Path(args.dest).expanduser()),
        "start": fmt_ts(start),
        "end": fmt_ts(end),
        "files": [record.to_json() for record in records],
        "count": len(records),
        "bytes": sum(record.bytes for record in records),
        "copied": sum(1 for record in records if record.copied),
        "checksums": str(checksums),
        "coverage": window_coverage(Path(args.log_dir).expanduser(), start, end),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Archived {payload['count']} file(s), {payload['bytes'] / 1e6:.1f} MB "
            f"({payload['copied']} newly copied, {payload['count'] - payload['copied']} already present)"
        )
        print(f"Checksums: {payload['checksums']}")
        for record in records:
            print(
                f"  {record.name:<44} {record.bytes:>10}  {record.verification}"
                f"{'' if record.copied else '  (existing copy verified)'}"
            )
    return 0


def _cmd_run_window(args: argparse.Namespace) -> int:
    result = run_window(
        name=args.name,
        window_start=parse_ts(args.start),
        window_end=parse_ts(args.end),
        output_dir=Path(args.output_dir).expanduser(),
        log_dir=Path(args.log_dir).expanduser(),
        analyzer=Path(args.analyzer).expanduser(),
        llama_log_dir=Path(args.llama_log_dir).expanduser() if args.llama_log_dir else None,
        manifest_path=Path(args.manifest).expanduser() if args.manifest else None,
        extra_args=args.extra_arg,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"[{result['name']}] exit={result['returncode']}")
        print(f"  command: {result['command']}")
        headline = result["headline"]
        if headline.get("sessions") is not None:
            print(
                f"  sessions={headline['sessions']} requests={headline['requests']} "
                f"total_tokens={headline['total_tokens']} "
                f"tokens_per_request={headline['tokens_per_request']}"
            )
        coverage = result["coverage"]
        print(
            f"  coverage: {coverage['measured_hours']} h of {coverage['window_hours']} h "
            f"({'complete' if coverage['complete'] else 'INCOMPLETE'})"
        )
        if result.get("manifest"):
            print(f"  manifest: {result['manifest']}")
    return 0 if result["returncode"] == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="proxy_usage_window_archive.py",
        description=(
            "Archive raw proxy log segments covering an evidence window, verify the copies, "
            "and regenerate the window with the proxy-usage-analysis CLI while recording a "
            "reproducible manifest."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser("coverage", help="report which files cover a window")
    coverage.add_argument("--log-dir", required=True)
    coverage.add_argument("--start", required=True, help="window start 'YYYY-MM-DD HH:MM:SS'")
    coverage.add_argument("--end", required=True, help="window end 'YYYY-MM-DD HH:MM:SS'")
    coverage.add_argument("--max-gap-seconds", type=float, default=3600.0)
    coverage.add_argument("--json", action="store_true")
    coverage.set_defaults(func=_cmd_coverage)

    archive = subparsers.add_parser("archive", help="copy + verify the raw segments for a window")
    archive.add_argument("--log-dir", required=True)
    archive.add_argument("--dest", required=True)
    archive.add_argument("--start", required=True)
    archive.add_argument("--end", required=True)
    archive.add_argument("--llama-log-dir", help="directory for llama-server.log* (defaults to --log-dir)")
    archive.add_argument("--allow-empty", action="store_true", help="succeed even when nothing is retained")
    archive.add_argument("--json", action="store_true")
    archive.set_defaults(func=_cmd_archive)

    run = subparsers.add_parser("run-window", help="regenerate one window and record the manifest block")
    run.add_argument("--name", required=True, help="window name, e.g. post24")
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--log-dir", default="/var/log/llama-proxy")
    run.add_argument("--llama-log-dir")
    run.add_argument("--analyzer", default=str(DEFAULT_ANALYZER))
    run.add_argument("--manifest", help="manifest file to append the window block to")
    run.add_argument("--extra-arg", action="append", default=[], help="extra CLI argument (repeatable)")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=_cmd_run_window)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
