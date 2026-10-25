#!/usr/bin/env bash
# Install this repository's committed git hooks (idempotent).
#
# Points `core.hooksPath` at the committed `.githooks/` directory so the
# committed hooks actually run — notably `.githooks/pre-push`, which refuses a
# push to `dev`/`main` unless the full test suite is green
# (LP-0MVFFJWPW007ZEPK; see docs/dev/git-hooks.md).
#
# The hooks are version-controlled files and are NEVER copied, overwritten or
# rewritten here: only the repository-local git config and the hooks'
# executable bit are touched. Re-running is safe and is the intended recovery
# path after a fresh clone or a wiped `.git/config`.
#
# `core.hooksPath` lives in the shared git config, so one run covers the main
# checkout and every linked worktree of this clone.
#
# Usage:
#   scripts/install_git_hooks.sh [--repo-root PATH] [--check] [--quiet]
#
#   --check   verify the install instead of performing it (exit 1 when the
#             hooks are not active, with the reason on stderr)
#   --quiet   only report problems
#
# Environment:
#   PI_SKIP_HOOK_INSTALL=1   skip the install (exit 0, no config change)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="install"
QUIET=0
HOOKS_PATH=".githooks"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="${2:?--repo-root needs a path}"
      shift 2
      ;;
    --check)
      MODE="check"
      shift
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "install_git_hooks: unknown argument '$1'" >&2
      echo "  Try: $(basename "$0") --help" >&2
      exit 2
      ;;
  esac
done

say() { [ "$QUIET" = "1" ] || echo "$@"; }

if [ ! -f "$REPO_ROOT/$HOOKS_PATH/pre-push" ]; then
  echo "install_git_hooks: $REPO_ROOT/$HOOKS_PATH/pre-push not found." >&2
  echo "  Run this from a checkout of this repository (or pass --repo-root)." >&2
  exit 1
fi

current="$(git -C "$REPO_ROOT" config --get core.hooksPath || true)"

verify() {
  local problem=0
  if [ "$current" != "$HOOKS_PATH" ]; then
    echo "install_git_hooks: core.hooksPath is '${current:-<unset>}', expected '$HOOKS_PATH'." >&2
    echo "  Fix: scripts/install_git_hooks.sh" >&2
    problem=1
  fi
  if [ ! -x "$REPO_ROOT/$HOOKS_PATH/pre-push" ]; then
    echo "install_git_hooks: $HOOKS_PATH/pre-push is not executable." >&2
    echo "  Fix: chmod +x $HOOKS_PATH/pre-push" >&2
    problem=1
  fi
  return "$problem"
}

if [ "$MODE" = "check" ]; then
  if verify; then
    say "install_git_hooks: git hooks are active (core.hooksPath = $HOOKS_PATH)"
    exit 0
  fi
  exit 1
fi

if [ "${PI_SKIP_HOOK_INSTALL:-}" = "1" ]; then
  say "install_git_hooks: skipped (PI_SKIP_HOOK_INSTALL=1)"
  exit 0
fi

if [ "$current" = "$HOOKS_PATH" ]; then
  say "install_git_hooks: core.hooksPath already set to $HOOKS_PATH (no change)"
else
  git -C "$REPO_ROOT" config core.hooksPath "$HOOKS_PATH"
  say "install_git_hooks: core.hooksPath: '${current:-<unset>}' -> $HOOKS_PATH"
fi

# git refuses to run a hook that is not executable. The committed mode should
# already be 0755; repair a checkout that lost it (e.g. a copied tree).
for hook in "$REPO_ROOT/$HOOKS_PATH"/*; do
  [ -f "$hook" ] || continue
  [ -x "$hook" ] || chmod +x "$hook"
done

current="$(git -C "$REPO_ROOT" config --get core.hooksPath || true)"
if ! verify; then
  exit 1
fi

say "install_git_hooks: done — committed hooks are active for this clone and all its worktrees."
say "  Verify:  git config --get core.hooksPath      # prints $HOOKS_PATH"
say "  Details: docs/dev/git-hooks.md"
