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
   the suite is red. The gate first verifies that it is measuring the
   **pushed commit** — the pushed `local_sha` must be the checkout's `HEAD`
   and the tree must be clean, otherwise the push is refused before the suite
   runs (see [The gate verifies the pushed
   commit](#the-gate-verifies-the-pushed-commit)). Feature-branch pushes never
   run the suite (fast loop; the full suite first runs at this integration
   gate, per SA-0MVECQUGF0064U81).
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

### The gate verifies the pushed commit

A green gate is only worth something if it is evidence about **the commit
being pushed**. The gate therefore measures the tree the push runs in (the
checkout's toplevel, which `implement.py finish` makes the main checkout)
only when that tree **is** the pushed commit:

- the checkout's `HEAD` must equal the pushed `local_sha` of every gated ref
  (`refs/heads/dev`, `refs/heads/main`) in this push; and
- the tree must be **clean** — no modified, staged or untracked paths.

Anything else is refused **before the suite runs**, because the runner keys
its cache on the git state (`HEAD` plus the content of every dirty path): a
stale or dirty checkout has a *different but valid* fingerprint, so a green
run there would be a legitimate cache hit for a tree the pushed commit never
was. The failure mode this closes: `implement.py finish` cannot fast-forward
the main checkout (it is dirty from concurrent work), warns, and pushes from
the stale checkout — the gate then went green in seconds and
`run_tests.py --summary` reported `missing` for the pushed commit
(LP-0MVSJUM3O003C0H9, observed on 2026-10-28).

Two refusals, both loud and actionable:

```text
ERROR: pre-push gate refuses this push: the tree it would measure is not the pushed commit.
  pushed commit  (refs/heads/dev): 903497fb36a2bc153d26533af8c2b56913703b09
  checked-out HEAD:            4e1691b4e168c758470e93aa578c904afbd8a92b
  gate tree:                   /home/…/llm
  …Recovery: bring the main checkout to the pushed commit and re-run
            'implement.py finish' …, or verify the commit in a worktree that is
            checked out at <pushed sha> and push from there.
```

```text
ERROR: pre-push gate refuses this push: the gate tree is dirty at the pushed commit.
  pushed commit: 50167e0098f7ae503b5056ca1a49a9219a621b7c
  gate tree:     /home/…/llm
  2 path(s) uncommitted, first 10:
     M proxy/tests/test_llama_local_status.py
     ?? scratch-untracked.txt
  …A full-suite run here fingerprints the dirty state, so a green result
  would not be evidence about the pushed commit; the push is refused
  before the suite runs.
```

**Recovery.** Bring the gate tree to the pushed commit and re-push:

1. clean the main checkout — commit or stash the changes (never discard
   someone else's work), and let `implement.py finish` fast-forward `dev` to
   the pushed commit; then
2. re-run `python3 ~/.pi/agent/skills/implement/scripts/implement.py finish
   <work-item-id>` (idempotent: it re-verifies commit → ancestry → status);
3. or, when the main checkout cannot be cleaned, verify in a worktree that
   **is** checked out at the pushed commit and push from there — the gate
   measures the worktree, which is clean and at the pushed commit, so the
   run is genuine evidence and its cache entry is mirrored into the main
   checkout's `.worklog/cache`:

   ```bash
   cd .worklog/worktrees/wl-<id>-<slug>
   git fetch origin dev && git rebase origin/dev
   WORKLOG_SKIP_PRE_PUSH=1 git push origin HEAD:refs/heads/dev
   ```

   (`WORKLOG_SKIP_PRE_PUSH=1` only skips the worklog sync, which cannot run
   from a linked worktree — the full-suite gate still runs.)

The guard is scoped to the gated refs: feature-branch pushes still skip the
suite, and an all-zero (deletion) `local_sha` carries no commit and is not a
mismatch. `.worklog/` is gitignored here, so worklog churn never makes the
tree look dirty. **No expected-dirty paths are declared** for this repo; if a
permanently dirty, gate-irrelevant path is ever needed, that is a deliberate
change to the hook, not a configuration knob.

**Escape hatches** stay explicit and loud: `TEST_SCOPE_SKIP=1` bypasses the
whole `dev`/`main` gate (the guard included) and `git push --no-verify`
skips every hook. Neither is silent — a bypassed push leaves no full-suite
cache entry, so post-push verification reports `missing` and the audit has no
evidence to consume.

> **Bootstrap note.** The push that *introduces* a new hook version runs the
> previous hook (the hook is read from the pushing checkout before the push
> lands), so the first post-landing push is the first one the guard protects.
> Also note `implement.py finish` pushes from the main checkout, so a skipped
> fast-forward now surfaces as this refusal rather than a silent green gate.

## Simulate a `dev` push without a remote

The hook reads the pushed refs on stdin, one line per ref, in git's order
(`<local-ref> <local-sha> <remote-ref> <remote-sha>`). From a **clean**
checkout whose `HEAD` is the sha you pass as `<local-sha>` (expect exit 0
and a full-suite run):

```bash
sha="$(git rev-parse HEAD)"
printf 'refs/heads/wl-x %s refs/heads/dev %s\n' "$sha" \
  0000000000000000000000000000000000000000 \
  | sh .githooks/pre-push origin git@example.invalid:repo.git
echo "exit=$?"   # expect: 0
```

The same invocation with a `<local-sha>` that is **not** the checkout's
`HEAD` demonstrates the pushed-commit guard (expect exit 1 and no suite):

```bash
stale="$(git rev-parse HEAD^)"
printf 'refs/heads/wl-x %s refs/heads/dev %s\n' "$stale" \
  0000000000000000000000000000000000000000 \
  | sh .githooks/pre-push origin git@example.invalid:repo.git
echo "exit=$?"   # expect: 1, with both shas and the gate tree in the message
```

With a deliberately red tree (or `TEST_SCOPE_SKIP` unset and a failing
`TESTS` fixture in a sandbox) the same invocation exits non-zero and refuses
the push — that is the regression guard verified by
`tests/test_git_hooks.py`.

## Hook matrix

| Hook | Purpose |
|---|---|
| `pre-push` | branch policy, context budget, pushed-commit guard + full-suite gate on `dev`/`main`, worklog sync |
| `post-checkout` | worklog sync after branch checkout (main checkout only) |
| `post-merge` | worklog sync after a pull/merge; delegates to the repo-local `.githooks/worklog-post-pull` (main checkout only) |
| `post-rewrite` | worklog sync after an amend/rebase; delegates to the repo-local `.githooks/worklog-post-pull` (main checkout only, stdin forwarded) |
| `worklog-post-pull` | the post-pull sync implementation the two wrappers above delegate to |

All non-`pre-push` hooks are worklog plumbing and skip inside linked
worktrees and temp sync worktrees.

## The hook files are repo-owned

Everything under `.githooks/` belongs to **this** repository: the hook bodies
are committed here and delegate only to scripts committed here. `post-merge`
and `post-rewrite` are thin wrappers that resolve their delegate from the
repository root
(`exec "$(git rev-parse --show-toplevel)/.githooks/worklog-post-pull" "$@"`),
so they keep working when the checkout moves and cannot silently break because
another project was renamed or deleted. Their marker line
(`worklog:post-pull-hook:v2`) makes a stale `v1` wrapper — the template copies
that pointed at the `llm-manager` checkout and emitted
`exec: ... not found` on every merge — identifiable at a glance.

Never point a hook at another checkout's path, and never generate or copy hook
bodies from a template: `scripts/install_git_hooks.sh` deliberately touches only
`core.hooksPath` and the executable bit, so a broken hook body can only be
fixed in the committed file itself.
