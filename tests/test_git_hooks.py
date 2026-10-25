#!/usr/bin/env python3
"""Verification tests for the committed git hooks and their installer.

Covers LP-0MVFFJWPW007ZEPK: this checkout must actually *run* its committed
``.githooks/pre-push`` gate (``core.hooksPath``), and that gate must run the
full test suite when a push targets ``dev``/``main`` while a feature-branch
push stays fast.

Every test drives a throwaway sandbox repository (a copy of ``.githooks/`` and
``scripts/install_git_hooks.sh``) so nothing here touches the real checkout's
git config or executes the real suite. The hook is invoked the way git invokes
it: as an executable with the pushed refs on stdin.

Assertions are behavioural — exit code of the hook, whether the runner ran at
all, and the arguments it received — never a grep of the hook source.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = Path(".githooks/pre-push")
INSTALLER = Path("scripts/install_git_hooks.sh")
ZERO_SHA = "0" * 40

#: Fake runner: records its invocation and exits with a scripted status. It
#: also stands in for the real runner's cache write (an entry under the git
#: dir's ``test-cache``), which the hook mirrors to the main checkout.
FAKE_RUNNER = '''#!/usr/bin/env python3
import os
import subprocess
import sys

marker = os.environ.get("PRE_PUSH_TEST_MARKER")
if marker:
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\\n")

cache_entry = os.environ.get("PRE_PUSH_TEST_CACHE_ENTRY")
if cache_entry:
    git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    os.makedirs(os.path.join(git_dir, "test-cache", cache_entry), exist_ok=True)

sys.exit(int(os.environ.get("PRE_PUSH_TEST_EXIT", "0")))
'''


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout


def _sandbox(tmp_path: Path) -> Path:
    """Create a throwaway git repo carrying this repo's hooks + installer."""
    root = tmp_path / "sandbox"
    root.mkdir()
    shutil.copytree(REPO_ROOT / ".githooks", root / ".githooks")
    (root / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / INSTALLER, root / INSTALLER)

    _git(root, "init", "-q", "-b", "feat", ".")
    _git(root, "config", "user.name", "Hook Test")
    _git(root, "config", "user.email", "hook-test@example.invalid")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox baseline")
    return root


def _install_runner(root: Path, exit_code: int = 0) -> Path:
    """Install a repo-local fake runner, as a framework checkout would."""
    runner = root / "skill/test/scripts/run_tests.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        FAKE_RUNNER.replace('os.environ.get("PRE_PUSH_TEST_EXIT", "0")', str(exit_code)),
        encoding="utf-8",
    )
    return runner


def _hook(
    cwd: Path,
    remote_ref: str,
    *,
    tmp_path: Path,
    home: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the pre-push hook once, with *remote_ref* as the push target."""
    sha = _git(cwd, "rev-parse", "HEAD").strip()
    stdin = f"refs/heads/feat {sha} {remote_ref} {ZERO_SHA}\n"
    fake_home = home or (tmp_path / "home")
    fake_home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(fake_home),
        # Keep the worklog sync out of these tests: it is not what is under
        # test and would need a worklog store.
        "WORKLOG_SKIP_PRE_PUSH": "1",
        "PRE_PUSH_TEST_MARKER": str(tmp_path / "runner-calls.log"),
    }
    env.update(extra_env or {})
    return subprocess.run(
        ["sh", str(cwd / HOOK), "origin", "git@example.invalid:repo.git"],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _marker(tmp_path: Path) -> str:
    path = tmp_path / "runner-calls.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _install(root: Path, *extra: str, env: dict[str, str] | None = None):
    run_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(root.parent / "home"),
    }
    run_env.update(env or {})
    return subprocess.run(
        ["bash", str(root / INSTALLER), *extra],
        cwd=root,
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
    )


# ---------------------------------------------------------------------------
# AC1/AC4 — idempotent installer activates the committed hooks
# ---------------------------------------------------------------------------


def test_installer_activates_hooks_path_idempotently(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    hook_before = (root / HOOK).read_bytes()

    first = _install(root)
    assert first.returncode == 0, first.stderr
    assert _git(root, "config", "--get", "core.hooksPath").strip() == ".githooks"

    second = _install(root)
    assert second.returncode == 0, second.stderr
    assert _git(root, "config", "--get", "core.hooksPath").strip() == ".githooks"
    # Re-running must never overwrite (or delete) the committed hooks.
    assert (root / HOOK).read_bytes() == hook_before

    check = _install(root, "--check")
    assert check.returncode == 0, check.stderr


def test_installer_check_fails_when_hooks_not_installed(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)

    check = _install(root, "--check")

    assert check.returncode != 0
    assert "install_git_hooks.sh" in check.stderr


def test_installer_env_bypass_leaves_config_untouched(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)

    result = _install(root, env={"PI_SKIP_HOOK_INSTALL": "1"})

    assert result.returncode == 0
    unset = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unset.returncode != 0, "PI_SKIP_HOOK_INSTALL must not set core.hooksPath"


# ---------------------------------------------------------------------------
# AC2 — dev/main pushes run the full suite and a red suite refuses the push
# ---------------------------------------------------------------------------


def test_dev_push_runs_full_suite_gate(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr
    calls = _marker(tmp_path)
    assert "--scope full" in calls
    assert "--strict-git-env" in calls


def test_main_push_runs_full_suite_gate(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)

    result = _hook(root, "refs/heads/main", tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--scope full" in _marker(tmp_path)


def test_red_suite_refuses_dev_push(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=1)

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path)

    assert result.returncode != 0
    assert "Full test suite failed" in result.stderr
    assert "TEST_SCOPE_SKIP=1" in result.stderr


def test_test_scope_skip_bypasses_the_gate(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=1)

    result = _hook(
        root, "refs/heads/dev", tmp_path=tmp_path, extra_env={"TEST_SCOPE_SKIP": "1"}
    )

    assert result.returncode == 0, result.stderr
    assert _marker(tmp_path) == "", "the runner must not be invoked when bypassed"


# ---------------------------------------------------------------------------
# AC3 — feature-branch pushes skip the suite entirely
# ---------------------------------------------------------------------------


def test_feature_branch_push_skips_suite(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    # A runner that would fail the push if it were invoked at all.
    _install_runner(root, exit_code=1)

    result = _hook(root, "refs/heads/feat", tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr
    assert _marker(tmp_path) == "", "feature-branch pushes must not run the suite"


# ---------------------------------------------------------------------------
# AC2 — runner resolution: repo-local first, then the global skill install
# ---------------------------------------------------------------------------


def test_global_skill_runner_is_used_when_no_repo_local_runner(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    home = tmp_path / "home"
    global_runner = home / ".pi/agent/skills/test/scripts/run_tests.py"
    global_runner.parent.mkdir(parents=True)
    global_runner.write_text(FAKE_RUNNER, encoding="utf-8")

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path, home=home)

    assert result.returncode == 0, result.stderr
    assert "--scope full" in _marker(tmp_path), "the global runner was not invoked"


def test_repo_local_runner_takes_precedence(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    home = tmp_path / "home"
    global_runner = home / ".pi/agent/skills/test/scripts/run_tests.py"
    global_runner.parent.mkdir(parents=True)
    global_runner.write_text(
        FAKE_RUNNER + "\nraise SystemExit(9)  # must never run\n", encoding="utf-8"
    )

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path, home=home)

    assert result.returncode == 0, result.stderr
    assert "--scope full" in _marker(tmp_path)


def test_missing_runner_fails_closed_on_dev_push(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path)

    assert result.returncode != 0
    assert "no test runner" in result.stderr.lower()
    assert "TEST_SCOPE_SKIP=1" in result.stderr


def test_missing_runner_does_not_block_feature_branch_push(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)

    result = _hook(root, "refs/heads/feat", tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Cache visibility — a worktree gate run must be visible to the main checkout
# ---------------------------------------------------------------------------


def test_worktree_run_mirrors_cache_entry_into_main_checkout(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    (root / ".worklog/cache").mkdir(parents=True)
    _install(root)
    worktree = tmp_path / "wt"
    _git(root, "worktree", "add", "-q", "-b", "wt-branch", str(worktree))
    _install_runner(worktree, exit_code=0)

    result = _hook(
        worktree,
        "refs/heads/dev",
        tmp_path=tmp_path,
        extra_env={"PRE_PUSH_TEST_CACHE_ENTRY": "entry-from-worktree"},
    )

    assert result.returncode == 0, result.stderr
    assert (root / ".worklog/cache/entry-from-worktree").is_dir(), (
        "the worktree gate run's cache entry was not mirrored into the main "
        "checkout, so post-push verification and audits would miss it"
    )
