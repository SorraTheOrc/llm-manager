#!/usr/bin/env python3
"""Pytest wrapper around the sandboxed ``start-proxy.sh`` detach harness.

``tests/test_start_proxy_detach.sh`` is a standalone bash harness (the
convention shared with ``tests/test_start_proxy_keys.sh`` and
``tests/test_start_proxy_mode.sh``). No ``tests/*.sh`` file is collected by the
default pytest suite, so this thin wrapper runs the harness as a subprocess to
make the detach-survival regression test part of that suite — the same pattern
as ``tests/test_git_hooks.py`` and ``tests/test_rebuild_and_restart_mtp.py``.

Covers AC3 of LP-0MVGT0E6W000Q8CY via the test-first child
LP-0MVK3K4IL0055DEQ: the launched server must survive a ``SIGKILL`` of the
launcher's entire process group (a foreground ``exec`` of uvicorn does not).

GREEN phase — the detach launch now lives in ``start-proxy.sh``
(LP-0MVK3KIPZ002SR32), so the harness must pass on its own. It was written
first (child LP-0MVK3K4IL0055DEQ) under a ``strict`` xfail marker that failed
hard if the fix landed without clearing it; that marker is gone now that the
fix is in, and this wrapper's success is the regression gate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path("tests/test_start_proxy_detach.sh")


def test_detach_survival_harness_passes() -> None:
    """Run the sandboxed shell harness; it must exit 0 with no failures."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available, cannot run the detach harness")

    result = subprocess.run(
        [bash, str(REPO_ROOT / HARNESS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        "tests/test_start_proxy_detach.sh failed "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
