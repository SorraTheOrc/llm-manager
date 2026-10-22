# Git hooks: the dev-push full-suite gate

This repository ships **committed git hooks** in `.githooks/`. They only run
when the checkout is configured to use them (`core.hooksPath = .githooks`).
The most important one is `.githooks/pre-push`, which enforces, on every
push:

1. **Branch policy** — pushes to `main`/`master`/`HEAD` are refused;
   work happens in `wl-<id>-<desc>` feature branches and is integrated via
   `dev`.
2. **Context-budget regression gate** — the pi session startup context
   surface must stay within the committed thresholds
   (`docs/dev/context-budget.thresholds.json`, measured by the framework's
   `context-audit` skill). The gate is fail-open when the measuring tool is
   not installed.
   *Thresholds were re-pinned on 2026-10-22* (LP-0MVFFJWPW007ZEPK): the file
   still pinned `total: 967` B while this repo's `AGENTS.md` had grown to
   3487 B, so the gate would have refused **every** `dev` push the moment it
   was activated. It now pins the approved surface (`project_agents` /
   `total` = 4096 B, measured 4040 B). Growing `AGENTS.md` past the pinned
   budget fails the push until the budget is deliberately re-pinned with the
   measurement recorded — that is the gate working, not a defect.
3. **Full-suite test gate on `dev`/`main` pushes** — a push whose target ref
   is `refs/heads/dev` or `refs/heads/main` runs the **full test suite**
   (`run_tests.py --scope full --strict-git-env`) and is **refused** while
   the suite is red. Feature-branch pushes never run the suite (fast loop;
   the full suite first runs at this integration gate, per
   SA-0MVECQUGF0064U81).
4. **Worklog sync** — the worklog data branch is synced before the code
   push (skipped when pushing from a linked worktree).

Without activation **none** of this runs: this is what
[LP-0MVFFJWPW007ZEPK](worklog) fixed — before it, `git config core.hooksPath`
was unset and `.git/hooks/pre-push` absent, so a red tree could reach `dev`.

## Install (one-time, per clone)

```bash
scripts/install_git_hooks.sh
```

That is idempotent and safe to re-run (it only sets `core.hooksPath` and
repairs the hooks' executable bit, never copies or rewrites the committed
hook files). `core.hooksPath` is **common config**, so one run covers the
main checkout and **every linked worktree** of the clone. A fresh clone or a
wiped `.git/config` drops it — the first command an agent/operator should
try when a push behaves as if no hook ran is the installer again.

Bypass (for legitimate one-off cases; not recommended):

```bash
PI_SKIP_HOOK_INSTALL=1 scripts/install_git_hooks.sh   # don't touch config
git push ... --no-verify                               # skip ALL hooks
```

## Verify

```bash
git config --get core.hooksPath        # expect: .githooks
scripts/install_git_hooks.sh --check   # exit 0 when active, lists the fix otherwise
ls -l .githooks/pre-push               # must be executable
```

## What happens on a push to `dev`

The gate resolves the test runner **repo-local first**
(`skill/test/scripts/run_tests.py`, framework-style checkouts), then the
global pi skill install (`~/.pi/agent/skills/test/scripts/run_tests.py` —
what this repo uses, since it ships no repo-local `skill/` tree). With no
runner at all the push **fails closed** on a `dev`/`main` target: a machine
that cannot run the suite cannot sneak a red tree into `dev`. Escape hatch:
`TEST_SCOPE_SKIP=1` (loud, not recommended).

The runner writes its per-git-state results to `<repo>/.worklog/cache` (or,
in a worktree — no `.worklog` — the worktree's `<main>/.git/worktrees/<name>/test-cache`).
To keep the evidence readable by the main checkout's post-push verification
and the audit skill, a worktree gate run mirrors its new cache entries into
the main checkout's `.worklog/cache`.

**Timeout:** the cold full suite measured **~302 s** on this host
(2026-10-22, 4.3k tests), which is close enough to the 600 s default
`implement.py finish` push timeout that host load could tip a cold push over
it. This repo therefore provisions headroom in `.pi/test-config.json`
(`timeoutPerCommand: 1800`, read both by `run_tests.py`'s per-command timeout
and by `implement.py finish`'s push step). After the suite is cached (2 h
TTL, git-state fingerprint) repeat pushes at the same commit are cheap.

## Simulate a `dev` push without a remote

The hook reads the pushed refs on stdin
(`<local-ref> <local-sha> <remote-ref> <remote-sha>`). From a checkout with
a green tree (expect exit 0, full suite runs):

```bash
printf 'refs/heads/wl-x refs/heads/x 0000000000000000000000000000000000000000 refs/heads/dev 0000000000000000000000000000000000000000\n' \
  | sh .githooks/pre-push origin git@example.invalid:repo.git
```

With a deliberately red tree (or `TEST_SCOPE_SKIP` unset and a failing
`TESTS` fixture in a sandbox) the same invocation exits non-zero and refuses
the push — that is the regression guard verified by
`tests/test_git_hooks.py`.

## Hook matrix

| Hook | Purpose |
|---|---|
| `pre-push` | branch policy, context budget, full-suite gate on `dev`/`main`, worklog sync |
| `post-checkout` | worklog sync after branch checkout (main checkout only) |
| `post-merge`, `post-rewrite`, `worklog-post-pull` | worklog sync after pulls/merges (main checkout only) |

All non-`pre-push` hooks are worklog plumbing and skip inside linked
worktrees and temp sync worktrees.
