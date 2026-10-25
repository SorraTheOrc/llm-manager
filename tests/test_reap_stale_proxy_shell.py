#!/usr/bin/env python3
"""Pytest wrapper around the stale-proxy-twin reaper harness.

``tests/test_reap_stale_proxy.sh`` spawns stand-in processes whose command
lines match the real ``uvicorn proxy.server:app ... --port N`` shape (holding a
log file open, like the wedged 2026-10-23 twin) and asserts that
``proxy/scripts/reap-stale-proxy.sh`` terminates only the stale twin for the
requested port. No ``tests/*.sh`` file is collected by the default pytest
suite, so this thin wrapper makes the regression part of that suite — the same
convention as ``tests/test_start_proxy_detach_shell.py``.

Covers AC2 of LP-0MVLHMAWV008EKTI: exactly one proxy process may own the log,
and a bound-less twin is reaped instead of being left alive with an open
write descriptor.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path("tests/test_reap_stale_proxy.sh")


def test_reap_stale_proxy_harness_passes() -> None:
    """Run the sandboxed shell harness; it must exit 0 with no failures."""
    bash = shutil.which("bash")
    if bash is None:
        import pytest

        pytest.skip("bash is not available, cannot run the reaper harness")
    if shutil.which("pgrep") is None:
        import pytest

        pytest.skip("pgrep is not available, cannot run the reaper harness")

    result = subprocess.run(
        [bash, str(REPO_ROOT / HARNESS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        "tests/test_reap_stale_proxy.sh failed "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
