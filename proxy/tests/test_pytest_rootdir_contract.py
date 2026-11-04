"""Contract: pytest resolves the repository root as its rootdir for mixed selections.

Regression test for LP-0MVZ43UR8004IL8Y.

The changed-scope selection can mix ``.pi/skills/proxy-usage-analysis/tests`` with
``proxy/tests`` in a single pytest process. When ``proxy/pytest.ini`` was the only
pytest configuration file in the tree, pytest adopted it as the session config and
resolved ``rootdir`` to ``<repo>/proxy``. Test modules outside that root then lost the
file component of their node ids (``::Class::test``), so unrelated modules shared a
node-id prefix and module-level autouse fixtures leaked across them: the
``_no_prometheus`` fixture from ``test_capacity_signals.py`` replaced
``gpu_busy.compute_gpu_busy`` inside ``test_gpu_busy.py``, and three GPU-busy tests
failed with ``None`` because their ``patch("gpu_busy.query_prometheus")`` had nothing
left to intercept.

The behaviour under test is process-level (a pytest invocation is a process), so these
tests drive pytest in a subprocess rather than re-implementing its rootdir logic.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# One repository-tree file plus one skill-tree file is enough to exercise the
# rootdir selection: the two live on different sides of the ``proxy/`` boundary.
MIXED_SELECTION = (
    "proxy/tests/test_monitoring_deployment.py",
    ".pi/skills/proxy-usage-analysis/tests/test_gpu_busy.py",
)

# The exact mix that failed: the skill modules involved in the fixture leak plus a
# repo-tree module that used to pull rootdir into ``proxy/``.
LEAK_SELECTION = (
    "proxy/tests/test_monitoring_deployment.py",
    ".pi/skills/proxy-usage-analysis/tests/test_capacity_signals.py",
    ".pi/skills/proxy-usage-analysis/tests/test_gpu_busy.py",
)


def _pytest(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _nodeid_files(output: str) -> set[str]:
    """Return the file component of every collected node id in ``output``."""
    files = set()
    for line in output.splitlines():
        stripped = line.strip()
        if "::" not in stripped or stripped.startswith("="):
            continue
        files.add(stripped.split("::", 1)[0])
    return files


def test_mixed_selection_reports_the_repository_root_as_rootdir():
    proc = _pytest("--collect-only", *MIXED_SELECTION)
    match = re.search(r"^rootdir:\s*(.+?)\s*$", proc.stdout, re.MULTILINE)
    assert match, proc.stdout
    assert Path(match.group(1)).resolve() == REPO_ROOT, (
        "pytest resolved rootdir to "
        f"{match.group(1)!r}; it must be the repository root so node ids stay "
        "file-qualified across the proxy/ boundary"
    )


def test_mixed_selection_keeps_file_qualified_node_ids():
    proc = _pytest("--collect-only", "-q", *MIXED_SELECTION)
    files = _nodeid_files(proc.stdout)
    assert files, proc.stdout
    for relative in sorted(files):
        assert (REPO_ROOT / relative).is_file(), (
            f"node id file component {relative!r} does not resolve against the "
            "repository root — pytest picked a nested rootdir and stripped the path"
        )


def test_mixed_skill_and_proxy_selection_runs_without_cross_module_fixture_leaks():
    proc = _pytest("-q", "--disable-warnings", *LEAK_SELECTION)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
