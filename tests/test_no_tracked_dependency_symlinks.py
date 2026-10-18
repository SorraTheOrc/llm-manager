#!/usr/bin/env python3
"""Regression guard: no node_modules entry may be tracked by git (LP-0MV81ZIL8008RS9J).

The implement framework auto-symlinks the main checkout's node_modules into
worktrees, and `proxy/node_modules` was accidentally committed four times as
a broken self-referential symlink (LP-0MTCZ35X7009IZKE, LP-0MUIEK47X000WL2C,
LP-0MUI56OCI008EGML) despite .gitignore hardening (LP-0MUIH7FYS0010MLT). A
tracked symlink breaks `npm install` and Playwright runs inside `proxy/`.

These tests assert the repository's git state so that any future re-commit of
a node_modules path fails the suite loudly instead of shipping silently.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_no_node_modules_path_is_tracked() -> None:
    """AC1 guard: `git ls-files` must contain no node_modules entries."""
    tracked = _git("ls-files").splitlines()
    offenders = sorted(
        path
        for path in tracked
        if "node_modules" in Path(path).parts  # exact path component, not a filename substring
    )
    assert offenders == [], (
        f"git tracks node_modules path(s): {offenders}. A tracked node_modules "
        "symlink from the implement framework's worktree auto-symlink breaks "
        "`npm install`/Playwright in proxy/. Remove with `git rm --cached "
        "<path>` (LP-0MV81ZIL8008RS9J)."
    )


def test_proxy_node_modules_is_gitignored() -> None:
    """AC2 guard: proxy/node_modules must match a .gitignore rule.

    The repository-root .gitignore and proxy/.gitignore both ignore it (no
    trailing slash, so directories AND symlinks match). If this ever stops
    matching, a plain `git add -A` in a worktree can re-commit the framework's
    auto-symlinked node_modules.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", "proxy/node_modules"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "proxy/node_modules is not covered by a .gitignore rule — the "
        "implement framework symlinks it into worktrees and it could be "
        "re-committed (LP-0MV81ZIL8008RS9J).\n" + result.stderr
    )
    assert "node_modules" in result.stdout, (
        "proxy/node_modules matched an unexpected rule:\n" + result.stdout
    )
