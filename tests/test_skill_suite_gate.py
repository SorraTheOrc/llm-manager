#!/usr/bin/env python3
"""Drift guard: the resolved full-suite command set must cover the skill tests.

Parent LP-0MVOBJGB4001TQ8X found that the repo's dev-push full-suite gate
never executed the project-local skill tests under
``.pi/skills/proxy-usage-analysis/tests`` (529 tests): pytest's default
``norecursedirs`` includes ``.*``, so a recursive ``pytest`` run never
traversed ``.pi/``. The enforcement fix belongs upstream (SA-0MVQJSGLY0098DVB,
the test skill's ``run_tests.py`` ``full_suite_commands()``), which now emits
an extra explicit-path pytest command for hidden skill test directories. This
module is the drift guard (parent AC5, child LP-0MVT26OX9006WLCZ): it asserts
the **resolved** command set — the one the gate and the audit actually consume
— still carries that command, so a future configuration or resolution change
can never silently re-blind the gate.

The runner is resolved exactly like ``.githooks/pre-push``: repo-local
``skill/test/scripts/run_tests.py`` first, then the global pi skill install
``~/.pi/agent/skills/test/scripts/run_tests.py``. The test imports that module
through ``importlib`` and calls only ``full_suite_commands()`` — no nested
suite is ever executed, and a resolved run takes milliseconds. When neither
runner exists the module skips with a reason naming the expected install path,
so a documented limitation stays visible in the report instead of silently
passing (AC5, D-P3).

Below: the SA-0MVQJSGLY0098DVB rule and the "no ``suiteCommands`` pin" rule
(docs/dev/git-hooks.md) — the two things this file protects.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Repo-relative directory the guard must stay covered by the full-suite set.
SKILL_TEST_DIR = ".pi/skills/proxy-usage-analysis/tests"

#: The canonical repo pytest command (also the full-suite cache command key).
CANONICAL_PYTEST_CMD = "pytest -q -r a --disable-warnings"

#: Resolver parity with ``.githooks/pre-push`` — repo-local first, then the
#: global pi skill install. This repo ships no repo-local ``skill/`` tree, so
#: the global install is what the gate uses.
REPO_LOCAL_RUNNER = REPO_ROOT / "skill/test/scripts/run_tests.py"
GLOBAL_RUNNER = Path.home() / ".pi/agent/skills/test/scripts/run_tests.py"

_MISSING_RUNNER_REASON = (
    "resolved full-suite runner (run_tests.py) not found — expected at "
    f"{REPO_LOCAL_RUNNER} (repo-local) or {GLOBAL_RUNNER} (global pi skill "
    "install), exactly as resolved by .githooks/pre-push. The drift guard "
    "cannot assert the command set without it; this skip is the documented "
    "limitation D-P3 and stays visible in the report rather than silently "
    "passing."
)


def _resolve_runner(project_root: Path) -> Path | None:
    """Mirror ``.githooks/pre-push`` runner resolution.

    Repo-local ``skill/test/scripts/run_tests.py`` first, then the global pi
    skill install ``~/.pi/agent/skills/test/scripts/run_tests.py``. Returns
    ``None`` when neither exists (the pre-push hook fails closed then; the
    guard skips, see AC5).
    """
    if (project_root / "skill/test/scripts/run_tests.py").is_file():
        return project_root / "skill/test/scripts/run_tests.py"
    global_runner = Path.home() / ".pi/agent/skills/test/scripts/run_tests.py"
    return global_runner if global_runner.is_file() else None


@contextmanager
def _loaded_runner(project_root: Path) -> Iterator:
    """Import the resolved runner with ``importlib`` and yield its module.

    Only the module's import-time statements run (no suite is executed). The
    runner's own directory is put on ``sys.path`` for its sibling modules
    (``test_cache``, ``test_runner``); ``run_tests.py`` itself prepends the
    skills root for the ``shared.*`` package. Global state (``sys.path`` and
    newly registered ``sys.modules`` entries) is restored afterwards so the
    import cannot shadow stdlib or project modules for the rest of the suite.
    """
    runner_path = _resolve_runner(project_root)
    if runner_path is None:
        raise pytest.skip.Exception(_MISSING_RUNNER_REASON)

    path_before = list(sys.path)
    modules_before = set(sys.modules)
    module_name = f"_drift_guard_runner_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise pytest.skip.Exception(
            f"cannot import runner {runner_path} (importlib gave no spec/loader)"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        sys.path.insert(0, str(runner_path.parent))
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = path_before
        for name in set(sys.modules) - modules_before:
            del sys.modules[name]


def _full_suite_commands(project_root: Path) -> list[str]:
    """The resolved full-suite command set for *project_root*."""
    with _loaded_runner(project_root) as runner:
        commands = runner.full_suite_commands(project_root)
    return list(commands)


def _is_pytest_command(command: str) -> bool:
    tokens = shlex.split(command)
    return bool(tokens) and tokens[0] == "pytest"


def _target_commands(commands: list[str], path: str) -> list[str]:
    """Pytest commands that name *path* as an explicit (initial) path."""
    return [
        command
        for command in commands
        if _is_pytest_command(command) and path in shlex.split(command)
    ]


# ---------------------------------------------------------------------------
# AC1 — drift guard: the resolved full-suite command set covers the skill tests
# ---------------------------------------------------------------------------


def test_full_suite_command_set_targets_the_skill_test_dir() -> None:
    """Exactly one pytest command in the resolved set names the skill tests.

    SA-0MVQJSGLY0098DVB emits one extra explicit-path pytest command naming
    every ``.pi/skills/<name>/tests`` directory the recursive run cannot
    reach. If this assertion turns red, either that rule regressed (the
    upstream item is the owner) or this repo configured ``suiteCommands`` in
    ``.pi/test-config.json`` (the pinned list wins by design and silently
    drops the skill tests) — see docs/dev/git-hooks.md.
    """
    commands = _full_suite_commands(REPO_ROOT)
    targets = _target_commands(commands, SKILL_TEST_DIR)

    assert len(targets) == 1, (
        f"expected exactly one full-suite pytest command targeting "
        f"{SKILL_TEST_DIR!r}, found {len(targets)}: {commands!r}. "
        "The SA-0MVQJSGLY0098DVB rule (run_tests.py full_suite_commands) "
        "emits it; without it the dev-push gate silently drops ~529 skill "
        "tests (pytest's norecursedirs prunes .*/). Check for a "
        "suiteCommands pin in .pi/test-config.json — see docs/dev/git-hooks.md."
    )


# ---------------------------------------------------------------------------
# AC2 — additive, not a replacement: the canonical repo command stays present
# ---------------------------------------------------------------------------


def test_canonical_repo_pytest_command_is_still_present() -> None:
    """The canonical repo command remains, so the skill command is additive.

    The gate's full-suite cache entry for the repo tests is keyed on exactly
    this command (docs/dev/git-hooks.md); replacing it would silently reblind
    the repo's own 5 264-test suite, which is not what the upstream rule does.
    """
    commands = _full_suite_commands(REPO_ROOT)

    assert CANONICAL_PYTEST_CMD in commands, (
        f"the canonical repo pytest command {CANONICAL_PYTEST_CMD!r} is missing "
        f"from the resolved set {commands!r}. The skill-test command must be "
        "additive (SA-0MVQJSGLY0098DVB), never a replacement — check "
        "run_tests.py full_suite_commands() and any suiteCommands pin in "
        ".pi/test-config.json (docs/dev/git-hooks.md)."
    )


# ---------------------------------------------------------------------------
# AC3 — no pin: .pi/test-config.json must not carry suiteCommands
# ---------------------------------------------------------------------------


def test_no_suite_commands_pin_in_test_config() -> None:
    """A pinned ``suiteCommands`` would silently drop the skill tests.

    The pinned list wins by design: ``run_tests.py`` returns it verbatim and
    skips convention detection, which would remove the skill-test command and
    also disable changed-scope selection (the finish loop falls back to the
    full suite). The repo deliberately keeps only ``timeoutPerCommand``.
    """
    config_path = REPO_ROOT / ".pi/test-config.json"
    assert config_path.is_file(), (
        f"{config_path} is missing — the drift guard's no-pin assertion "
        "cannot evaluate. See docs/dev/git-hooks.md for the "
        "\u201cdo not pin suiteCommands\u201d rule."
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert "suiteCommands" not in config, (
        ".pi/test-config.json pins 'suiteCommands' — that pin silently drops "
        "the .pi/skills/*/tests skill tests (the pinned list wins by design) "
        "and also disables changed-scope selection (full-suite fallback on "
        "every implement.py finish). Remove the key; see docs/dev/git-hooks.md "
        "and SA-0MVQJSGLY0098DVB."
    )


# ---------------------------------------------------------------------------
# AC4 — resolver parity: repo-local first, then the global install; fast
# ---------------------------------------------------------------------------


def test_resolver_mirrors_pre_push_and_is_fast() -> None:
    """The runner resolves like ``.githooks/pre-push`` and runs in ms.

    This repo ships no repo-local ``skill/`` tree, so the resolved path must
    be the global pi skill install — the same path the hook invokes. The
    import-plus-resolve round trip must complete well under a second: the
    guard only ever calls ``full_suite_commands()`` after an import, never a
    nested suite (e.g. ``pytest --collect-only`` would take seconds).
    """
    assert not (REPO_ROOT / "skill").is_dir(), (
        "this repo is expected to ship no repo-local skill/ tree; if that "
        "changed, update the resolver-parity expectation with the pre-push hook"
    )

    with _loaded_runner(REPO_ROOT) as runner:
        resolved_path = Path(runner.__file__).resolve()
        assert resolved_path == GLOBAL_RUNNER.resolve(), (
            f"resolved runner {resolved_path} != global install "
            f"{GLOBAL_RUNNER} — resolver no longer mirrors .githooks/pre-push "
            "(repo-local first, then ~/.pi/agent/skills)."
        )

        start = time.monotonic()
        commands = runner.full_suite_commands(REPO_ROOT)
        elapsed = time.monotonic() - start

    assert isinstance(commands, list) and all(
        isinstance(c, str) for c in commands
    ), "full_suite_commands must return a list of command strings."
    assert elapsed < 1.0, (
        f"import + full_suite_commands took {elapsed:.3f}s — expected well "
        "under a second. The guard must never execute a nested suite (AC4)."
    )
    assert commands, "the resolved full-suite command set is empty."


# ---------------------------------------------------------------------------
# AC5 — explicit skip when no runner exists (documented limitation D-P3)
# ---------------------------------------------------------------------------


def test_skip_with_reason_when_no_runner(tmp_path: Path, monkeypatch) -> None:
    """Neither runner present ⇒ the guard skips, naming the expected path.

    ``pytest.skip`` keeps the absence visible in the report (an
    ``s``/``skipped`` row) instead of silently passing or hard-failing: a hard
    failure would break the suite on hosts without the pi test skill, a
    silent pass would hide a lost guard.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert _resolve_runner(REPO_ROOT) is None

    with pytest.raises(
        pytest.skip.Exception, match=r"\.pi/agent/skills/test/scripts/run_tests\.py"
    ):
        with _loaded_runner(REPO_ROOT):
            pytest.fail("_loaded_runner must skip, never load, with no runner")
