#!/usr/bin/env python3
"""Remediate work items spuriously closed under the wrong release version.

Incident LP-0MUEALIM60037F1S
============================

The ship wrapper (``run-release.js``) ran its post-release steps without a
release: an unrecognised flag was forwarded to ``merge-dev-to-main.sh``, which
printed usage and exited 0. The wrapper then accepted the *previous* tag
(``v0.1.18``) as the "released version" and invoked
``closeWorkItemsAfterRelease('0.1.18')``, closing 74 ``stage=in_review``
release-candidate items with the wrong reason
``Closed with reason: Shipped in v0.1.18`` — minutes before the real v0.1.19
release. The real release then found zero candidates, so ``CHANGELOG.md``
shipped an empty ``## v0.1.19`` section.

This tool repairs the damage idempotently:

1. Select close comments authored by ``worklog`` whose body is exactly
   ``Closed with reason: Shipped in v<bad_version>`` and whose ``createdAt``
   is on or after ``--since``. The timestamp guard is essential: the
   legitimate v0.1.18 close batch (2026-09-16) shares the same body and must
   never be touched.
2. Neutralise those comments — ``--mode delete`` (default) removes them;
   ``--mode neutralize`` rewrites the text to a corrective note.
3. Restore every affected item to ``completed`` / ``in_review`` (skip with
   ``--no-restore``).
4. Rewrite the ``## v<good_version>`` CHANGELOG section from the affected
   parent items, re-using the existing section's date (or today's when the
   section is new).
5. Close the affected top-level items with ``Shipped in v<good_version>``;
   ``wl close`` recursively closes their children, so the whole set ends
   ``done`` / ``completed`` with the correct reason. Parents already carrying
   a ``Shipped in v<good_version>`` comment are never re-closed.

Re-running after a successful sweep is a no-op: no spurious comments remain,
no items are restored, and the changelog is not rewritten.

Worklog sync caveat (why ``neutralize`` exists)
-----------------------------------------------

The worklog sync merges comments by **union of ids** — remote comments missing
locally are re-added (:func:`mergeComments` in the worklog sync store). A
hard ``wl comment delete`` is therefore not durable while the remote ref still
holds the comment: the next sync re-adds it (observed as
``Sync summary commentsAdded=<n>``). Comment *updates* are durable — the local
record wins for a shared id — so ``--mode neutralize`` is the way to make a
mislabeled comment stay corrected across syncs. ``--mode delete`` is retained
for worklogs without a sync remote.

Usage:

    python3 scripts/remediate_release_close_mislabel.py \\
        --bad-version 0.1.18 \\
        --good-version 0.1.19 \\
        --since 2026-09-23T00:00:00Z \\
        --mode neutralize \\
        --changelog CHANGELOG.md \\
        --close \\
        --dry-run
    # ... then drop --dry-run to apply.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

# ── Constants ────────────────────────────────────────────────────────────────

#: Author recorded by ``wl close`` on the close comment.
CLOSE_COMMENT_AUTHOR = "worklog"


def close_reason(version: str) -> str:
    """Return the ``wl close --reason`` argument for a release version."""
    return f"Shipped in v{version}"


def stored_close_comment(version: str) -> str:
    """Return the comment body ``wl close`` stores for a release version."""
    return f"Closed with reason: {close_reason(version)}"


def neutralized_body(
    bad_version: str,
    good_version: str | None,
    remediation_item_id: str | None = None,
) -> str:
    """Corrective replacement text for a spuriously-closed comment."""
    target = f"v{good_version}" if good_version else "the following release"
    suffix = f" (remediation {remediation_item_id})" if remediation_item_id else ""
    return (
        f"Close reason corrected{suffix}: this item shipped in {target}. "
        f"The earlier \"Shipped in v{bad_version}\" label was applied by a "
        "release post-step that ran without a release."
    )


#: Work item issue_type -> CHANGELOG category (mirrors generate-changelog.js).
ISSUE_TYPE_CATEGORY = {
    "feature": "feature",
    "bug": "bug",
    "chore": "other",
    "docs": "other",
    "task": "other",
    "epic": "other",
}

CATEGORY_ORDER = ("feature", "bug", "other")
CATEGORY_HEADINGS = {
    "feature": "### Features",
    "bug": "### Bug Fixes",
    "other": "### Other",
}


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkItem:
    """A work item as needed for remediation."""

    id: str
    title: str
    issue_type: str
    parent_id: str | None = None


@dataclass(frozen=True)
class Comment:
    """A work-item comment as needed for remediation."""

    id: str
    work_item_id: str
    author: str
    body: str
    created_at: str


@dataclass
class RemediationResult:
    """Outcome of a remediation run."""

    spurious_comment_ids: list[str] = field(default_factory=list)
    affected_item_ids: list[str] = field(default_factory=list)
    deleted_comment_ids: list[str] = field(default_factory=list)
    neutralized_comment_ids: list[str] = field(default_factory=list)
    restored_item_ids: list[str] = field(default_factory=list)
    closed_item_ids: list[str] = field(default_factory=list)
    changelog_updated: bool = False
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


class WorklogBoundary(Protocol):
    """Read/write boundary over Worklog (injectable for tests)."""

    def work_items(self) -> list[WorkItem]: ...

    def comments(self) -> list[Comment]: ...

    def delete_comment(self, comment_id: str) -> None: ...

    def update_comment(self, comment_id: str, body: str) -> None: ...

    def restore_item(self, item_id: str) -> None: ...

    def close_item(self, item_id: str, reason: str) -> None: ...


# ── Timestamp parsing ────────────────────────────────────────────────────────


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` accepted) to an aware UTC datetime."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# ── Selection ────────────────────────────────────────────────────────────────


def select_spurious_comments(
    comments: Sequence[Comment],
    *,
    bad_version: str,
    since: str,
) -> list[Comment]:
    """Select close comments produced by the buggy post-release close step.

    A comment is spurious when **all** of the following hold:

    - ``author`` is ``worklog`` (the only author ``wl close`` uses);
    - the body is exactly ``Closed with reason: Shipped in v<bad_version>``;
    - ``createdAt`` parses and is greater than or equal to ``since``.

    The version + timestamp guards together ensure the legitimate close batch
    from the *real* release of ``bad_version`` is never selected.
    """
    expected = stored_close_comment(bad_version)
    since_dt = parse_timestamp(since)
    if since_dt is None:
        raise ValueError(f"invalid --since timestamp: {since!r}")

    selected = []
    for comment in comments:
        if comment.author != CLOSE_COMMENT_AUTHOR:
            continue
        if comment.body != expected:
            continue
        comment_dt = parse_timestamp(comment.created_at)
        if comment_dt is None or comment_dt < since_dt:
            continue
        selected.append(comment)
    return selected


# ── CHANGELOG rendering ──────────────────────────────────────────────────────


def category_for(issue_type: str) -> str:
    """Map a Worklog ``issue_type`` to a CHANGELOG category."""
    return ISSUE_TYPE_CATEGORY.get((issue_type or "").strip().lower(), "other")


def build_release_section(
    version: str,
    date: str,
    items: Sequence[WorkItem],
) -> str:
    """Render a CHANGELOG release section grouped by issue-type category.

    Format matches ``generate-changelog.js`` with no LLM configured: each
    entry is ``- <title> (<id>)``.
    """
    grouped: dict[str, list[WorkItem]] = {category: [] for category in CATEGORY_ORDER}
    for item in items:
        grouped[category_for(item.issue_type)].append(item)

    lines = [f"## v{version} ({date})", ""]
    for category in CATEGORY_ORDER:
        entries = grouped[category]
        if not entries:
            continue
        lines.append(CATEGORY_HEADINGS[category])
        lines.append("")
        for item in entries:
            lines.append(f"- {item.title} ({item.id})")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


_SECTION_DATE_RE_TEMPLATE = r"^## v{version} \(([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})\)"


def extract_section_date(changelog_text: str, version: str) -> str | None:
    """Return the date recorded in an existing ``## v<version>`` section."""
    pattern = _SECTION_DATE_RE_TEMPLATE.format(version=re.escape(version))
    match = re.search(pattern, changelog_text, re.MULTILINE)
    return match.group(1) if match else None


def replace_release_section(changelog_text: str, version: str, new_section: str) -> str:
    """Replace the ``## v<version>`` section, preserving its position.

    When the section does not yet exist the new section is inserted directly
    after the top-level ``# Changelog`` heading block.
    """
    marker = f"## v{version}"
    start = changelog_text.find(marker)
    if start == -1:
        header_end = changelog_text.find("\n\n")
        if header_end == -1:
            return changelog_text.rstrip("\n") + "\n\n" + new_section + "\n"
        return (
            changelog_text[: header_end + 2]
            + new_section
            + "\n\n"
            + changelog_text[header_end + 2 :]
        )

    next_start = changelog_text.find("\n## v", start)
    if next_start == -1:
        return changelog_text[:start] + new_section + "\n"
    next_start += 1  # point at the leading '#' of the next section
    return changelog_text[:start] + new_section + "\n\n" + changelog_text[next_start:]


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# ── Remediation ──────────────────────────────────────────────────────────────


def remediate(
    boundary: WorklogBoundary,
    *,
    bad_version: str,
    good_version: str | None = None,
    since: str,
    mode: str = "delete",
    changelog_path: Path | None = None,
    restore: bool = True,
    close: bool = False,
    remediation_item_id: str | None = None,
    dry_run: bool = False,
    log=print,
) -> RemediationResult:
    """Run the remediation sweep against *boundary*.

    ``mode`` selects how spurious comments are neutralised: ``delete`` removes
    them; ``neutralize`` rewrites their text (durable across worklog syncs).
    """
    if mode not in ("delete", "neutralize"):
        raise ValueError(f"invalid mode: {mode!r} (expected 'delete' or 'neutralize')")

    result = RemediationResult(dry_run=dry_run)

    items = {item.id: item for item in boundary.work_items()}
    comments = boundary.comments()

    spurious = select_spurious_comments(
        comments, bad_version=bad_version, since=since
    )
    result.spurious_comment_ids = [comment.id for comment in spurious]

    if not spurious:
        log("No spurious close comments found — nothing to do (idempotent no-op).")
        return result

    affected_ids = sorted({comment.work_item_id for comment in spurious})
    result.affected_item_ids = affected_ids
    affected_items = [items[item_id] for item_id in affected_ids if item_id in items]
    parents = [item for item in affected_items if item.parent_id is None]

    # Parents already closed under the good version must not be re-closed
    # (re-running a partially-applied sweep must not duplicate comments).
    already_good = {
        comment.work_item_id
        for comment in comments
        if good_version
        and comment.author == CLOSE_COMMENT_AUTHOR
        and comment.body == stored_close_comment(good_version)
    }

    log(
        f"Found {len(spurious)} spurious close comment(s) across "
        f"{len(affected_ids)} item(s) ({len(parents)} parent(s)) "
        f"[mode={mode}]."
    )

    replacement = neutralized_body(bad_version, good_version, remediation_item_id)
    for comment in spurious:
        if dry_run:
            continue
        if mode == "neutralize":
            boundary.update_comment(comment.id, replacement)
            result.neutralized_comment_ids.append(comment.id)
        else:
            boundary.delete_comment(comment.id)
            result.deleted_comment_ids.append(comment.id)

    if restore:
        for item in affected_items:
            if dry_run:
                continue
            boundary.restore_item(item.id)
            result.restored_item_ids.append(item.id)

    if changelog_path is not None and good_version:
        path = Path(changelog_path)
        existing = path.read_text(encoding="utf-8") if path.is_file() else "# Changelog\n\n"
        date = extract_section_date(existing, good_version) or _today_utc()
        section = build_release_section(good_version, date, parents)
        updated = replace_release_section(existing, good_version, section)
        if updated != existing:
            if not dry_run:
                path.write_text(updated, encoding="utf-8")
            result.changelog_updated = True

    if close and good_version:
        for item in parents:
            if item.id in already_good:
                continue
            if dry_run:
                continue
            boundary.close_item(item.id, close_reason(good_version))
            result.closed_item_ids.append(item.id)

    return result


# ── Default Worklog boundary ─────────────────────────────────────────────────


class WlBoundary:
    """Default boundary backed by the ``wl`` CLI (single export for reads)."""

    def __init__(self, wl_cmd: str = "wl", export_dir: Path | None = None):
        self._wl = wl_cmd
        self._export_dir = export_dir
        self._items: list[WorkItem] | None = None
        self._comments: list[Comment] | None = None

    def _run(self, args: Sequence[str]) -> str:
        proc = subprocess.run(
            [self._wl, *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"wl {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout

    def _load(self) -> None:
        if self._items is not None:
            return
        with tempfile.TemporaryDirectory(dir=self._export_dir) as tmp:
            export_file = Path(tmp) / "worklog-export.jsonl"
            self._run(["export", "--file", str(export_file)])
            items: list[WorkItem] = []
            comments: list[Comment] = []
            for raw_line in export_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = entry.get("data", entry)
                if entry.get("type") == "comment" or (
                    data.get("workItemId") is not None and data.get("author") is not None
                ):
                    work_item_id = data.get("workItemId")
                    if work_item_id is None:
                        continue
                    comments.append(
                        Comment(
                            id=data.get("id", ""),
                            work_item_id=work_item_id,
                            author=data.get("author", ""),
                            body=data.get("comment", "") or "",
                            created_at=data.get("createdAt", "") or "",
                        )
                    )
                elif data.get("id") and data.get("title") is not None:
                    items.append(
                        WorkItem(
                            id=data["id"],
                            title=data.get("title", ""),
                            issue_type=data.get("issueType", "") or "",
                            parent_id=data.get("parentId"),
                        )
                    )
            self._items = items
            self._comments = comments

    def work_items(self) -> list[WorkItem]:
        self._load()
        return list(self._items or [])

    def comments(self) -> list[Comment]:
        self._load()
        return list(self._comments or [])

    def delete_comment(self, comment_id: str) -> None:
        self._run(["comment", "delete", comment_id, "--json"])

    def update_comment(self, comment_id: str, body: str) -> None:
        self._run(["comment", "update", comment_id, "--comment", body, "--json"])

    def restore_item(self, item_id: str) -> None:
        self._run(
            ["update", item_id, "--status", "completed", "--stage", "in_review", "--json"]
        )

    def close_item(self, item_id: str, reason: str) -> None:
        self._run(["close", item_id, "--force", "--reason", reason, "--json"])


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remediate work items spuriously closed under the wrong release "
            "version (LP-0MUEALIM60037F1S)."
        )
    )
    parser.add_argument("--bad-version", required=True, help="Wrong release version, e.g. 0.1.18")
    parser.add_argument(
        "--good-version",
        help="Correct release version, e.g. 0.1.19 (required to rewrite/close)",
    )
    parser.add_argument(
        "--since",
        required=True,
        help="ISO-8601 timestamp; only comments created at/after this are spurious",
    )
    parser.add_argument(
        "--mode",
        choices=("delete", "neutralize"),
        default="delete",
        help="How to neutralise spurious comments (default: delete)",
    )
    parser.add_argument("--changelog", help="Path to CHANGELOG.md to rewrite")
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not restore affected items to completed/in_review",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close affected top-level items with the good version",
    )
    parser.add_argument(
        "--remediation-item",
        help="Work item id to cite in neutralized comment text",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without mutating")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.close or args.changelog) and not args.good_version:
        parser.error("--good-version is required with --close or --changelog")
    if args.mode == "neutralize" and not args.good_version:
        parser.error("--good-version is required with --mode neutralize")

    boundary = WlBoundary()
    try:
        result = remediate(
            boundary,
            bad_version=args.bad_version,
            good_version=args.good_version,
            since=args.since,
            mode=args.mode,
            changelog_path=Path(args.changelog) if args.changelog else None,
            restore=not args.no_restore,
            close=args.close,
            remediation_item_id=args.remediation_item,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    payload = {
        "dryRun": result.dry_run,
        "spuriousComments": len(result.spurious_comment_ids),
        "affectedItems": len(result.affected_item_ids),
        "deletedComments": len(result.deleted_comment_ids),
        "neutralizedComments": len(result.neutralized_comment_ids),
        "restoredItems": len(result.restored_item_ids),
        "closedItems": len(result.closed_item_ids),
        "changelogUpdated": result.changelog_updated,
        "errors": result.errors,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            "Remediation {mode}: {spuriousComments} spurious comment(s), "
            "{affectedItems} affected item(s), {deletedComments} deleted, "
            "{neutralizedComments} neutralized, {restoredItems} restored, "
            "{closedItems} closed, changelog {changelog}{errors}".format(
                mode="(dry-run)" if result.dry_run else "complete",
                changelog="updated" if result.changelog_updated else "unchanged",
                errors=(
                    f", {len(result.errors)} error(s)" if result.errors else ""
                ),
                **payload,
            )
        )
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
