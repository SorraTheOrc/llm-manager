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

The pushed-commit guard suite (LP-0MVSJUM3O003C0H9) rides on the same sandbox:
the gate must verify the commit it is pushing, so a tree that is *not* the
pushed commit (stale ``HEAD``) or a tree that is dirty at it never produces a
green gate. The refusals are asserted alongside the controls that must keep
working (clean tree, feature branch, deletion push, escape hatch).
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
FAKE_RUNNER = """#!/usr/bin/env python3
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
"""


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    )
    return result.stdout


#: Repository overrides that would redirect a fixture git command at another
#: repository (or another repository's index).
_GIT_OVERRIDE_VARS = frozenset(
    {
        "GIT_INDEX_FILE",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    }
)


def _git_env() -> dict[str, str]:
    """A hermetic environment for the sandbox's own git commands.

    ``implement.py finish`` exports a temporary ``GIT_INDEX_FILE`` while it
    runs the suite it is gating. An inherited value would make the sandbox's
    ``git add``/``git commit`` write into the *gate's* index, so the run could
    not be attributed to the would-be-commit tree and the gate rejects it.
    A fixture therefore drops the repository overrides.
    """
    return {name: value for name, value in os.environ.items() if name not in _GIT_OVERRIDE_VARS}


def _sandbox(tmp_path: Path) -> Path:
    """Create a throwaway git repo carrying this repo's hooks + installer."""
    root = tmp_path / "sandbox"
    root.mkdir()
    shutil.copytree(REPO_ROOT / ".githooks", root / ".githooks")
    (root / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / INSTALLER, root / INSTALLER)
    # Mirror the real checkout: worklog state (the suite cache included) is
    # gitignored, so worklog churn can never make the gate tree look dirty.
    (root / ".gitignore").write_text(".worklog/\n", encoding="utf-8")
    (root / "README.md").write_text("sandbox\n", encoding="utf-8")

    _git(root, "init", "-q", "-b", "feat", ".")
    _git(root, "config", "user.name", "Hook Test")
    _git(root, "config", "user.email", "hook-test@example.invalid")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox baseline")
    return root


def _install_runner(root: Path, exit_code: int = 0) -> Path:
    """Install a repo-local fake runner, as a framework checkout would.

    Committed, because a real repo-local runner is tracked: the gate only
    measures a tree that is clean at the pushed commit, so an untracked runner
    would turn every green-path test into a dirty-tree refusal.
    """
    runner = root / "skill/test/scripts/run_tests.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        FAKE_RUNNER.replace('os.environ.get("PRE_PUSH_TEST_EXIT", "0")', str(exit_code)),
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox: add fake runner")
    return runner


def _commit_all(root: Path, message: str) -> str:
    """Commit everything pending in *root* and return the new ``HEAD``."""
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _stale_sha(root: Path) -> str:
    """Advance the sandbox by one commit; return the sha that is now behind.

    Models the skipped pre-push fast-forward: the pushed commit (the returned
    sha) is no longer the commit the checkout has on disk.
    """
    stale = _git(root, "rev-parse", "HEAD").strip()
    (root / "LATER.txt").write_text("later commit\n", encoding="utf-8")
    _commit_all(root, "sandbox: later commit")
    return stale


def _make_dirty(root: Path) -> tuple[str, str]:
    """Leave uncommitted work behind: a modified tracked file and an untracked one."""
    (root / "README.md").write_text("uncommitted edit\n", encoding="utf-8")
    (root / "scratch-untracked.txt").write_text("scratch\n", encoding="utf-8")
    return "README.md", "scratch-untracked.txt"


def _hook(
    cwd: Path,
    remote_ref: str,
    *,
    tmp_path: Path,
    home: Path | None = None,
    extra_env: dict[str, str] | None = None,
    local_sha: str | None = None,
    remote_sha: str | None = None,
    extra_stdin_lines: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the pre-push hook once, with *remote_ref* as the push target.

    ``local_sha`` defaults to the checkout's ``HEAD``\u2014 the only shape git
    produces from a clean checkout. Pass another commit's sha to model the
    skipped pre-push fast-forward (``implement.py finish`` pushing a commit the
    checkout is no longer at).
    """
    sha = local_sha if local_sha is not None else _git(cwd, "rev-parse", "HEAD").strip()
    remote = remote_sha if remote_sha is not None else ZERO_SHA
    stdin = f"refs/heads/feat {sha} {remote_ref} {remote}\n"
    for line in extra_stdin_lines or []:
        stdin += line + "\n"
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

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path, extra_env={"TEST_SCOPE_SKIP": "1"})

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
    global_runner.write_text(FAKE_RUNNER + "\nraise SystemExit(9)  # must never run\n", encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Pushed-commit guard — the gate verifies the commit it is pushing
# (LP-0MVSJUM3O003C0H9, producer Q1=A repo-local / Q3=B strict)
# ---------------------------------------------------------------------------


def test_stale_dev_push_is_refused(tmp_path: Path) -> None:
    """A skipped fast-forward must never be gated against the stale checkout."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    stale = _stale_sha(root)
    head = _git(root, "rev-parse", "HEAD").strip()

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path, local_sha=stale)

    assert result.returncode != 0, "gating a tree that is not the pushed commit must be refused"
    assert stale in result.stderr, "the refusal must name the pushed commit"
    assert head in result.stderr, "the refusal must name the checked-out HEAD"
    assert str(root) in result.stderr, "the refusal must name the tree it would measure"
    assert "implement.py finish" in result.stderr, "the refusal must carry the recovery step"
    assert _marker(tmp_path) == "", "the runner must never run for another tree"


def test_stale_main_push_is_refused(tmp_path: Path) -> None:
    """``main`` is gated on the same pushed-commit terms as ``dev``."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    stale = _stale_sha(root)
    head = _git(root, "rev-parse", "HEAD").strip()

    result = _hook(root, "refs/heads/main", tmp_path=tmp_path, local_sha=stale)

    assert result.returncode != 0
    assert stale in result.stderr
    assert head in result.stderr
    assert _marker(tmp_path) == ""


def test_dirty_tree_at_matching_sha_is_refused(tmp_path: Path) -> None:
    """A dirty fingerprint is never evidence about the pushed commit."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    modified, untracked = _make_dirty(root)

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path)

    assert result.returncode != 0, "a dirty gate tree must be refused"
    assert modified in result.stderr, "the refusal must list the dirty paths"
    assert untracked in result.stderr
    assert _marker(tmp_path) == ""


def test_conflicting_gated_shas_in_one_push_are_refused(tmp_path: Path) -> None:
    """One tree cannot be the pushed commit for two different gated shas."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    stale = _stale_sha(root)
    head = _git(root, "rev-parse", "HEAD").strip()

    result = _hook(
        root,
        "refs/heads/dev",
        tmp_path=tmp_path,
        extra_stdin_lines=[f"refs/heads/feat2 {stale} refs/heads/main {ZERO_SHA}"],
    )

    assert result.returncode != 0
    assert stale in result.stderr
    assert head in result.stderr
    assert _marker(tmp_path) == ""


def test_clean_tree_at_matching_sha_runs_full_suite(tmp_path: Path) -> None:
    """The green path is unchanged: clean tree at the pushed commit runs the suite."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr
    calls = _marker(tmp_path)
    assert "--scope full" in calls
    assert "--strict-git-env" in calls


def test_feature_branch_with_mismatched_sha_still_skips_suite(tmp_path: Path) -> None:
    """The guard is scoped to dev/main: the feature-branch fast loop is untouched."""
    root = _sandbox(tmp_path)
    # A runner that would refuse the push if it were invoked at all.
    _install_runner(root, exit_code=1)
    stale = _stale_sha(root)

    result = _hook(root, "refs/heads/feat", tmp_path=tmp_path, local_sha=stale)

    assert result.returncode == 0, result.stderr
    assert _marker(tmp_path) == ""


def test_deletion_push_on_dev_is_not_a_sha_mismatch(tmp_path: Path) -> None:
    """An all-zero (deletion) local sha carries no commit, so it is not a mismatch."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=0)
    remote = _git(root, "rev-parse", "HEAD").strip()

    result = _hook(root, "refs/heads/dev", tmp_path=tmp_path, local_sha=ZERO_SHA, remote_sha=remote)

    assert result.returncode == 0, result.stderr
    assert "--scope full" in _marker(tmp_path), "the gate must still run for a deletion push"


def test_test_scope_skip_bypasses_the_pushed_commit_guard(tmp_path: Path) -> None:
    """The documented escape hatch bypasses the guard as well as the gate."""
    root = _sandbox(tmp_path)
    _install_runner(root, exit_code=1)
    stale = _stale_sha(root)

    result = _hook(
        root,
        "refs/heads/dev",
        tmp_path=tmp_path,
        local_sha=stale,
        extra_env={"TEST_SCOPE_SKIP": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert _marker(tmp_path) == ""


# ---------------------------------------------------------------------------
# post-merge / post-rewrite delegation (LP-0MVK6XGOF000C45C)
#
# The two post-pull wrappers must delegate to *this* repository's
# ``.githooks/worklog-post-pull``, never to a hard-coded path in another
# checkout. The wrappers shipped as template copies that pointed at the
# ``llm-manager`` checkout and printed ``exec: ... not found`` on every pull
# while the sync silently did not run. The helpers below keep this block
# self-contained so it never depends on the pre-push fixtures' shape, and the
# behaviours asserted are purely behavioural (delegate invoked, argv/stdin
# forwarded, stderr clean) — never a grep of the hook source.
# ---------------------------------------------------------------------------

POST_PULL_HOOKS = ("post-merge", "post-rewrite")
POST_PULL_DELEGATE = Path(".githooks/worklog-post-pull")

#: Fake ``.githooks/worklog-post-pull`` installed into the sandbox so the real
#: ``wl sync`` never runs in tests. Records argv and stdin, then exits with a
#: scripted status.
FAKE_POST_PULL = """#!/bin/sh
if [ -n "$POST_PULL_ARGS_MARKER" ]; then
  : > "$POST_PULL_ARGS_MARKER"
  for arg in "$@"; do
    printf '%s\\n' "$arg" >> "$POST_PULL_ARGS_MARKER"
  done
fi
if [ -n "$POST_PULL_STDIN_MARKER" ]; then
  cat > "$POST_PULL_STDIN_MARKER"
fi
exit "${POST_PULL_EXIT:-0}"
"""

#: The v1 wrapper body this repo shipped while it pointed at the
#: ``llm-manager`` template checkout. Used to prove the post-pull assertions
#: fail loudly when a hook delegates outside its own repository.
TEMPLATE_POST_PULL_HOOK = """#!/bin/sh
# worklog:post-pull-hook:v1
# Wrapper that delegates to central Worklog post-pull script (committed hooks).
exec "/nonexistent-llm-manager-checkout/.githooks/worklog-post-pull" "$@"
"""


def _install_fake_post_pull(root: Path) -> None:
    """Replace the sandbox post-pull delegate with a marker-writing fake."""
    delegate = root / POST_PULL_DELEGATE
    delegate.write_text(FAKE_POST_PULL, encoding="utf-8")
    delegate.chmod(0o755)


def _post_pull_env(tmp_path: Path) -> dict[str, str]:
    """Hermetic environment for the post-* hooks and their fake delegate.

    Repository overrides are dropped: ``implement.py finish`` exports a
    temporary ``GIT_INDEX_FILE`` while it gates the suite, and an inherited
    value would redirect the sandbox's git calls at the gate's index.
    """
    (tmp_path / "home").mkdir(exist_ok=True)
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env.update(
        {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(tmp_path / "home"),
            "POST_PULL_ARGS_MARKER": str(tmp_path / "post-pull-args.log"),
            "POST_PULL_STDIN_MARKER": str(tmp_path / "post-pull-stdin.log"),
        }
    )
    return env


def _delegate_args(tmp_path: Path) -> list[str]:
    path = tmp_path / "post-pull-args.log"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _delegate_stdin(tmp_path: Path) -> str:
    path = tmp_path / "post-pull-stdin.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _run_post_hook(
    root: Path, hook: str, *args: str, stdin: str = "", tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    """Run a post-* hook directly, the way git would (args + stdin)."""
    return subprocess.run(
        ["sh", str(root / ".githooks" / hook), *args],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        env=_post_pull_env(tmp_path),
        check=False,
    )


def _run_fixture_git(root: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the sandbox with the fake delegate's markers configured."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env=_post_pull_env(tmp_path),
        check=False,
    )


def _post_pull_case(tmp_path: Path, hook: str) -> Path:
    """A per-hook sandbox directory, so both wrappers run in one test."""
    case = tmp_path / hook
    case.mkdir(parents=True, exist_ok=True)
    return case


def test_post_pull_hooks_delegate_to_repo_local_script(tmp_path: Path) -> None:
    """Both wrappers run this repository's delegate with the hook's arguments."""
    for hook in POST_PULL_HOOKS:
        case = _post_pull_case(tmp_path, hook)
        root = _sandbox(case)
        _install_fake_post_pull(root)

        result = _run_post_hook(root, hook, "some-arg", "another-arg", tmp_path=case)

        assert result.returncode == 0, f"{hook}: {result.stderr}"
        assert _delegate_args(case) == ["some-arg", "another-arg"], hook
        assert result.stderr == "", f"{hook} stderr is not clean: {result.stderr!r}"
        assert "not found" not in result.stderr, hook


def test_post_pull_hooks_forward_stdin(tmp_path: Path) -> None:
    """post-rewrite's rewritten-commit list on stdin reaches the delegate."""
    payload = f"{'a' * 40} {'b' * 40}\n{'c' * 40} {'d' * 40}\n"
    for hook in POST_PULL_HOOKS:
        case = _post_pull_case(tmp_path, hook)
        root = _sandbox(case)
        _install_fake_post_pull(root)

        result = _run_post_hook(root, hook, "rebase", stdin=payload, tmp_path=case)

        assert result.returncode == 0, f"{hook}: {result.stderr}"
        assert _delegate_args(case) == ["rebase"], hook
        assert _delegate_stdin(case) == payload, hook


def test_post_pull_hook_pointing_outside_the_sandbox_fails_this_suite(
    tmp_path: Path,
) -> None:
    """Guard the guard: the v1 template wrapper must fail these assertions.

    A hook that delegates to an absolute path outside the repository runs no
    repo-local delegate and prints ``exec: ... not found``. This is exactly the
    defect LP-0MVK6XGOF000C45C fixes, and it must remain visible to the tests
    above (which assert the delegate ran and stderr stayed clean).
    """
    root = _sandbox(tmp_path)
    _install_fake_post_pull(root)
    stale = root / ".githooks/post-merge"
    stale.write_text(TEMPLATE_POST_PULL_HOOK, encoding="utf-8")
    stale.chmod(0o755)

    result = _run_post_hook(root, "post-merge", "0", tmp_path=tmp_path)

    assert result.returncode != 0, "a delegate outside the repository must fail"
    assert _delegate_args(tmp_path) == [], "no repo-local delegate should run"
    assert "not found" in result.stderr, "the v1 template hook must surface its broken delegation on stderr"


def test_git_fast_forward_merge_runs_repo_local_delegate(tmp_path: Path) -> None:
    """A real ``git merge --ff-only`` runs the sync with no shell noise."""
    root = _sandbox(tmp_path)
    _install(root)
    _install_fake_post_pull(root)
    _git(root, "checkout", "-q", "-b", "ff-source")
    (root / "ff.txt").write_text("ff\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "ff commit")
    _git(root, "checkout", "-q", "feat")

    result = _run_fixture_git(root, tmp_path, "merge", "--ff-only", "ff-source")

    assert result.returncode == 0, result.stderr
    assert "not found" not in result.stderr
    assert _delegate_args(tmp_path) == ["0"], "git passes the squash flag (0 = not a squash merge) to post-merge"


def test_git_commit_amend_runs_repo_local_delegate_with_rewritten_shas(
    tmp_path: Path,
) -> None:
    """A real ``git commit --amend`` drives post-rewrite, stdin forwarded."""
    root = _sandbox(tmp_path)
    _install(root)
    _install_fake_post_pull(root)
    old_sha = _git(root, "rev-parse", "HEAD").strip()
    (root / "amend.txt").write_text("amend\n", encoding="utf-8")
    _git(root, "add", "-A")

    result = _run_fixture_git(root, tmp_path, "commit", "-q", "--amend", "--no-edit")

    assert result.returncode == 0, result.stderr
    assert "not found" not in result.stderr
    new_sha = _git(root, "rev-parse", "HEAD").strip()
    assert new_sha != old_sha, "the amend must rewrite the commit"
    assert _delegate_args(tmp_path) == ["amend"]
    assert _delegate_stdin(tmp_path) == f"{old_sha} {new_sha}\n"
