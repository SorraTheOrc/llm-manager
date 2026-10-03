#!/usr/bin/env python3
"""Tests for scripts/remediate_release_close_mislabel.py (LP-0MUEALIM60037F1S).

The remediation tool repairs work items spuriously closed under the wrong
release version and restores the affected CHANGELOG section. These tests use
an injected fake Worklog boundary so the live worklog is never touched.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "remediate_release_close_mislabel.py"
)
_spec = importlib.util.spec_from_file_location(
    "remediate_release_close_mislabel", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses can resolve string annotations by module.
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

WorkItem = mod.WorkItem
Comment = mod.Comment


# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeBoundary:
    """In-memory WorklogBoundary capturing every mutation."""

    def __init__(self, items, comments):
        self._items = list(items)
        self._comments = list(comments)
        self.deleted = []
        self.updated = []
        self.restored = []
        self.closed = []

    def work_items(self):
        return list(self._items)

    def comments(self):
        return list(self._comments)

    def delete_comment(self, comment_id):
        self.deleted.append(comment_id)
        self._comments = [c for c in self._comments if c.id != comment_id]

    def update_comment(self, comment_id, body):
        self.updated.append((comment_id, body))
        self._comments = [
            Comment(c.id, c.work_item_id, c.author, body, c.created_at)
            if c.id == comment_id
            else c
            for c in self._comments
        ]

    def restore_item(self, item_id):
        self.restored.append(item_id)

    def close_item(self, item_id, reason):
        self.closed.append((item_id, reason))


def _mixed_dataset():
    """Realistic dataset with spurious, legitimate, and unrelated comments."""
    items = [
        WorkItem("LP-PARENT1", "Add startup ramp", "feature"),
        WorkItem("LP-CHILD1", "Startup ramp tests", "task", parent_id="LP-PARENT1"),
        WorkItem("LP-PARENT2", "Fix empty responses", "bug"),
    ]
    comments = [
        # Spurious: correct author + body, created after the cut-off.
        Comment("C1", "LP-PARENT1", "worklog", mod.stored_close_comment("0.1.18"), "2026-09-23T14:46:11.322Z"),
        Comment("C2", "LP-CHILD1", "worklog", mod.stored_close_comment("0.1.18"), "2026-09-23T14:46:11.331Z"),
        # Legitimate: same body, but from the real v0.1.18 release (before cut-off).
        Comment("C3", "LP-PARENT2", "worklog", mod.stored_close_comment("0.1.18"), "2026-09-16T09:20:00.000Z"),
        # Same window but a different (also wrong) release version.
        Comment("C4", "LP-PARENT2", "worklog", mod.stored_close_comment("0.1.17"), "2026-09-23T14:46:00.000Z"),
        # Same window/body but not authored by wl close.
        Comment("C5", "LP-PARENT2", "alice", mod.stored_close_comment("0.1.18"), "2026-09-23T14:46:00.000Z"),
        # Same window/body/author but a plain comment body variant.
        Comment("C6", "LP-PARENT2", "worklog", "Closed with reason: Shipped in v0.1.18 extra", "2026-09-23T14:46:00.000Z"),
    ]
    return items, comments


CHANGELOG_WITH_EMPTY_SECTION = """# Changelog

## v0.1.19 (2026-09-23)

## v0.1.18 (2026-09-16)
### Features
- Slot counts (LP-0MU03AL730000B5W)
"""


# ── Selection ────────────────────────────────────────────────────────────────


def test_select_spurious_comments_guards_author_version_and_time():
    _, comments = _mixed_dataset()
    selected = mod.select_spurious_comments(comments, bad_version="0.1.18", since="2026-09-23T00:00:00Z")
    assert [c.id for c in selected] == ["C1", "C2"]


def test_select_spurious_comments_cut_off_is_inclusive():
    comments = [
        Comment("A", "LP-X", "worklog", mod.stored_close_comment("0.1.18"), "2026-09-23T00:00:00.000Z"),
    ]
    selected = mod.select_spurious_comments(comments, bad_version="0.1.18", since="2026-09-23T00:00:00Z")
    assert [c.id for c in selected] == ["A"]


def test_select_spurious_comments_rejects_invalid_since():
    with pytest.raises(ValueError):
        mod.select_spurious_comments([], bad_version="0.1.18", since="not-a-date")


# ── CHANGELOG rendering ──────────────────────────────────────────────────────


def test_build_release_section_groups_parent_items_by_category():
    items = [
        WorkItem("LP-1", "Add a feature", "feature"),
        WorkItem("LP-2", "Fix a bug", "bug"),
        WorkItem("LP-3", "Chore work", "chore"),
        WorkItem("LP-4", "Epic work", "epic"),
    ]
    section = mod.build_release_section("0.1.19", "2026-09-23", items)
    assert section.startswith("## v0.1.19 (2026-09-23)\n")
    assert "### Features\n\n- Add a feature (LP-1)" in section
    assert "### Bug Fixes\n\n- Fix a bug (LP-2)" in section
    # chore and epic both map to "other"
    assert "### Other" in section
    assert "- Chore work (LP-3)" in section
    assert "- Epic work (LP-4)" in section


def test_build_release_section_omits_empty_categories():
    section = mod.build_release_section(
        "0.1.19", "2026-09-23", [WorkItem("LP-1", "Only feature", "feature")]
    )
    assert "### Features" in section
    assert "### Bug Fixes" not in section
    assert "### Other" not in section


def test_replace_release_section_replaces_existing_empty_section():
    new_section = mod.build_release_section(
        "0.1.19", "2026-09-23", [WorkItem("LP-1", "Add a feature", "feature")]
    )
    updated = mod.replace_release_section(CHANGELOG_WITH_EMPTY_SECTION, "0.1.19", new_section)
    assert updated.count("## v0.1.19") == 1
    assert "- Add a feature (LP-1)" in updated
    # The v0.1.18 section is untouched and still follows.
    assert "## v0.1.18 (2026-09-16)" in updated
    assert updated.index("## v0.1.19") < updated.index("## v0.1.18")


def test_replace_release_section_inserts_when_absent():
    base = "# Changelog\n\n## v0.1.18 (2026-09-16)\n### Features\n- Old (LP-OLD)\n"
    new_section = mod.build_release_section(
        "0.1.20", "2026-09-24", [WorkItem("LP-2", "New thing", "feature")]
    )
    updated = mod.replace_release_section(base, "0.1.20", new_section)
    assert updated.index("## v0.1.20") < updated.index("## v0.1.18")
    assert "- New thing (LP-2)" in updated


def test_extract_section_date_reads_existing_date_and_defaults_missing():
    assert mod.extract_section_date(CHANGELOG_WITH_EMPTY_SECTION, "0.1.19") == "2026-09-23"
    assert mod.extract_section_date(CHANGELOG_WITH_EMPTY_SECTION, "9.9.9") is None


# ── Remediation sweep ────────────────────────────────────────────────────────


def test_remediate_deletes_restores_rewrites_and_closes(tmp_path):
    items, comments = _mixed_dataset()
    boundary = FakeBoundary(items, comments)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_WITH_EMPTY_SECTION, encoding="utf-8")

    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        changelog_path=changelog,
        close=True,
    )

    # AC1: only the two spurious comments are deleted.
    assert boundary.deleted == ["C1", "C2"]
    # AC2: only the affected items are restored.
    assert boundary.restored == ["LP-CHILD1", "LP-PARENT1"]
    # AC4: only top-level items are closed (children close recursively).
    assert boundary.closed == [("LP-PARENT1", "Shipped in v0.1.19")]
    assert result.closed_item_ids == ["LP-PARENT1"]
    # AC3: the changelog section is populated from the affected parents.
    text = changelog.read_text(encoding="utf-8")
    assert "- Add startup ramp (LP-PARENT1)" in text
    assert "## v0.1.18 (2026-09-16)" in text
    assert result.changelog_updated is True


def test_remediate_is_idempotent(tmp_path):
    items, comments = _mixed_dataset()
    boundary = FakeBoundary(items, comments)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_WITH_EMPTY_SECTION, encoding="utf-8")

    kwargs = dict(
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        changelog_path=changelog,
        close=True,
    )
    first = mod.remediate(boundary, **kwargs)
    text_after_first = changelog.read_text(encoding="utf-8")
    second = mod.remediate(boundary, **kwargs)

    assert first.affected_item_ids
    assert second.spurious_comment_ids == []
    assert second.affected_item_ids == []
    assert boundary.deleted == ["C1", "C2"]  # no further deletions
    assert boundary.restored == ["LP-CHILD1", "LP-PARENT1"]  # no further restores
    assert boundary.closed == [("LP-PARENT1", "Shipped in v0.1.19")]  # no further closes
    assert changelog.read_text(encoding="utf-8") == text_after_first


def test_remediate_skips_reclosing_parents_already_good(tmp_path):
    """Re-running after the close landed but the comment was re-added must not
    duplicate the good close comment (idempotent close)."""
    items, comments = _mixed_dataset()
    # Simulate a partial application: spurious comments still present, but the
    # good close comment already landed on the parent.
    comments.append(
        Comment(
            "G1",
            "LP-PARENT1",
            "worklog",
            mod.stored_close_comment("0.1.19"),
            "2026-09-23T16:39:29.000Z",
        )
    )
    boundary = FakeBoundary(items, comments)
    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        close=True,
    )
    assert result.spurious_comment_ids == ["C1", "C2"]
    assert boundary.closed == []  # parent already good -> not re-closed


def test_remediate_neutralize_rewrites_comments_durably(tmp_path):
    items, comments = _mixed_dataset()
    boundary = FakeBoundary(items, comments)
    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        mode="neutralize",
        restore=False,
        remediation_item_id="LP-REM",
    )

    # Only the two spurious comments are rewritten; nothing is deleted.
    assert boundary.deleted == []
    assert [cid for cid, _ in boundary.updated] == ["C1", "C2"]
    assert result.neutralized_comment_ids == ["C1", "C2"]
    body = boundary.updated[0][1]
    assert "v0.1.19" in body and "v0.1.18" in body and "LP-REM" in body

    # Idempotent: the rewritten text no longer matches the spurious pattern.
    second = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        mode="neutralize",
        restore=False,
    )
    assert second.spurious_comment_ids == []
    assert len(boundary.updated) == 2


def test_remediate_rejects_unknown_mode():
    boundary = FakeBoundary(*_mixed_dataset())
    with pytest.raises(ValueError):
        mod.remediate(
            boundary,
            bad_version="0.1.18",
            since="2026-09-23T00:00:00Z",
            mode="explode",
        )


def test_remediate_no_restore_skips_restore(tmp_path):
    items, comments = _mixed_dataset()
    boundary = FakeBoundary(items, comments)
    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        since="2026-09-23T00:00:00Z",
        restore=False,
    )
    assert boundary.deleted == ["C1", "C2"]
    assert boundary.restored == []
    assert result.restored_item_ids == []


def test_remediate_dry_run_makes_no_mutations(tmp_path):
    items, comments = _mixed_dataset()
    boundary = FakeBoundary(items, comments)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_WITH_EMPTY_SECTION, encoding="utf-8")

    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        changelog_path=changelog,
        close=True,
        dry_run=True,
    )

    assert result.spurious_comment_ids == ["C1", "C2"]
    assert boundary.deleted == []
    assert boundary.restored == []
    assert boundary.closed == []
    assert changelog.read_text(encoding="utf-8") == CHANGELOG_WITH_EMPTY_SECTION


def test_remediate_noop_when_no_spurious_comments(tmp_path):
    boundary = FakeBoundary(
        [WorkItem("LP-PARENT", "Title", "feature")],
        [Comment("C", "LP-PARENT", "worklog", mod.stored_close_comment("0.1.18"), "2026-09-16T09:20:00Z")],
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG_WITH_EMPTY_SECTION, encoding="utf-8")

    result = mod.remediate(
        boundary,
        bad_version="0.1.18",
        good_version="0.1.19",
        since="2026-09-23T00:00:00Z",
        changelog_path=changelog,
        close=True,
    )

    assert result.spurious_comment_ids == []
    assert result.changelog_updated is False
    assert boundary.restored == []
    assert boundary.closed == []
    assert changelog.read_text(encoding="utf-8") == CHANGELOG_WITH_EMPTY_SECTION


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_requires_good_version_with_close_or_changelog(capsys):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--bad-version", "0.1.18", "--since", "2026-09-23T00:00:00Z", "--close"])
    assert exc.value.code == 2
    assert "--good-version" in capsys.readouterr().err
