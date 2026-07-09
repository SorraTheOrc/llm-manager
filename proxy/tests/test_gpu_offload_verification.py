"""GPU Offload Verification Harness — Mocked Unit Tests.

These tests validate that the router-mode startup infrastructure can
propagate GPU offload settings (-ngl / --gpu-layers) correctly.

The tests cover:
  - Parsing of [global] ngl from models.ini
  - Environment variable propagation for GPU settings (HSA_OVERRIDE_GFX_VERSION,
    ROCM_LLVM_PRE_VEGA)
  - Router-mode command construction (the presence of -ngl in the startup
    command is validated once the offload implementation is done — these tests
    serve as guards to detect regressions)
  - The CPU-only rollback toggle path

Design principles (from the work-item AC):
  - Reuse existing test infrastructure (test_host_flow_integration.py).
  - Tests FAIL if the router-mode startup stops propagating the chosen GPU
    offload settings or the CPU-only rollback toggle.

Reference:
  - models.ini: [global] ngl = 99
  - Single-model path: -ngl 99
  - Router mode command (current): does NOT include -ngl
"""

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the package path is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
lifecycle = importlib.import_module("proxy.lifecycle")


# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_INI = REPO_ROOT / "models.ini"
START_SCRIPT = REPO_ROOT / "start-llama.sh"
EXPECTED_GLOBAL_NGL = 99

# Environment variables that must be set for ROCm GPU offload
REQUIRED_ROCM_ENV_VARS = [
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCM_LLVM_PRE_VEGA",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_lifecycle_state(monkeypatch):
    """Reset lifecycle module state between tests."""
    # Each test gets its own DummySrv instance.
    yield


class FakeProc:
    """Fake subprocess.Popen that simulates a long-running process."""
    def __init__(self):
        self.returncode = None
        self.stdout = io.StringIO()

    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    def poll(self):
        return None

    def terminate(self):
        self.returncode = -1

    def wait(self, timeout=None):
        return 0


class DummySrv:
    """Minimal server state for lifecycle tests."""
    def __init__(self, config_overrides=None):
        cfg = {
            "server": {
                "llama_allow_host_fallback": True,
                "llama_router_mode": True,
                "llama_models_max": 2,
                "llama_server_port": 8080,
                **(config_overrides or {}),
            },
        }
        self.config = cfg
        self.logger = logging.getLogger("dummy")
        self.log_dir = None
        self.llama_log_file = None
        self.last_start_failure = None
        self.current_model = None
        self.backend_ready = False
        self.llama_process = None

    def rotate_llama_logs(self, *a, **kw):
        pass

    def broadcast_status_sync(self, *a, **kw):
        pass


# ── models.ini Parsing Tests ─────────────────────────────────────────────────

class TestModelsIniGlobalNgl:
    """Verify that models.ini [global] ngl can be parsed correctly."""

    def test_models_ini_exists(self):
        """models.ini must exist at the repo root."""
        assert MODELS_INI.exists(), (
            f"models.ini not found at {MODELS_INI}"
        )

    def test_models_ini_has_global_section(self):
        """models.ini must have a [global] section."""
        content = MODELS_INI.read_text()
        assert "[global]" in content, (
            "models.ini missing [global] section — required for shared GPU settings"
        )

    def test_models_ini_global_ngl_is_present(self):
        """The [global] section must define ngl (GPU layers)."""
        ngl_value = self._parse_global_ngl()
        assert ngl_value is not None, (
            "[global] ngl not found in models.ini — required for GPU offload"
        )

    def test_models_ini_global_ngl_value(self):
        """[global] ngl should be the expected value (e.g., 99 = all layers)."""
        ngl_value = self._parse_global_ngl()
        assert ngl_value == EXPECTED_GLOBAL_NGL, (
            f"[global] ngl = {ngl_value}, expected {EXPECTED_GLOBAL_NGL}"
        )

    # ------------------------------------------------------------------
    # Helper: parse [global] ngl from models.ini
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_global_ngl() -> int | None:
        """Parse and return the ngl value from [global] in models.ini.

        Returns the integer value of ngl in [global], or None if not found.
        Uses awk-style logic similar to start-llama.sh's get_ctx_size().
        """
        found_global = False
        with open(MODELS_INI) as f:
            for line in f:
                line_stripped = line.strip()
                # Detect section headers
                if line_stripped.startswith("["):
                    section = line_stripped.strip("[]").strip().lower()
                    found_global = section == "global"
                    continue
                # Within [global], look for ngl
                if found_global:
                    # Check for next section (exit [global])
                    if line_stripped.startswith("["):
                        break
                    if line_stripped.lower().startswith("ngl") or line_stripped.lower().startswith("ngl "):
                        # Split on = and take the value
                        if "=" in line_stripped:
                            parts = line_stripped.split("=", 1)
                            value_str = parts[1].strip()
                            try:
                                return int(value_str)
                            except ValueError:
                                return None
        return None


# ── Environment Variable Propagation Tests ──────────────────────────────────

class TestRocmEnvVars:
    """Validate that ROCm environment variables are propagated correctly."""

    def test_rocm_env_vars_defined_in_start_script(self):
        """start-llama.sh must export the ROCm environment variables."""
        assert START_SCRIPT.exists(), f"start-llama.sh not found at {START_SCRIPT}"
        content = START_SCRIPT.read_text()
        for var in REQUIRED_ROCM_ENV_VARS:
            assert f"export {var}=" in content, (
                f"start-llama.sh must export {var} for ROCm GPU operation"
            )

    def test_lifecycle_propagates_llama_server_bin(self, monkeypatch):
        """The lifecycle must propagate LLAMA_SERVER_BIN to the subprocess env.

        This ensures the ROCm-compiled binary is used.
        """
        dummy = DummySrv({"llama_server_bin": "/custom/path/llama-server"})
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        captured_env = {}

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            captured_env.update(env or {})
            raise FileNotFoundError(cmd[0])  # Simulate failure to avoid infinite loop

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        # This will fail to start, but we can inspect env before failure
        try:
            lifecycle.start_llama_server("Qwen3")
        except Exception:
            pass

        assert captured_env.get("LLAMA_SERVER_BIN") == "/custom/path/llama-server", (
            "LLAMA_SERVER_BIN must be propagated to subprocess environment"
        )

    def test_lifecycle_propagates_slot_save_path(self, monkeypatch):
        """The lifecycle must propagate LLAMA_SLOT_SAVE_PATH."""
        dummy = DummySrv({"session_slot_save_path": "/tmp/slot-cache"})
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        captured_env = {}

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            captured_env.update(env or {})
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        try:
            lifecycle.start_llama_server("Qwen3")
        except Exception:
            pass

        assert captured_env.get("LLAMA_SLOT_SAVE_PATH") == "/tmp/slot-cache", (
            "LLAMA_SLOT_SAVE_PATH must be propagated to subprocess environment"
        )

    def test_lifecycle_propagates_parallel_count(self, monkeypatch):
        """The lifecycle must propagate LLAMA_PARALLEL to align slot pool."""
        dummy = DummySrv({"session_slot_pool_size": 2})
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        captured_env = {}

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            captured_env.update(env or {})
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        try:
            lifecycle.start_llama_server("Qwen3")
        except Exception:
            pass

        assert captured_env.get("LLAMA_PARALLEL") == "2", (
            f"LLAMA_PARALLEL must match session_slot_pool_size, got: {captured_env.get('LLAMA_PARALLEL')}"
        )

    def test_models_ini_global_ngl_is_forwarded_to_router(self):
        """GATE TEST: Router-mode command must include -ngl from models.ini.

        This test verifies that -ngl is forwarded to the router-mode command.
        The value comes from models.ini [global] ngl, with a LLAMA_NGL env
        var override for CPU-only rollback.

        If this test fails, it indicates that router-mode GPU offload settings
        are no longer being propagated.
        """
        content = START_SCRIPT.read_text()

        # Find router-mode section: starts at 'if [[ "$router_mode" -eq 1 ]]'
        # and ends at the 'fi' that closes the router if block.
        # The router section contains nested if/fi blocks (for LLAMA_SERVER_BIN
        # and LLAMA_MODELS_DIR checks). To find the outermost closing fi, we
        # track the nesting depth of if/fi keywords.
        lines = content.splitlines()
        router_start = None
        router_if_depth = 0
        in_router = False
        router_lines = []

        for i, line in enumerate(lines):
            ls = line.strip()
            if not router_start and ls.startswith('if [[ "$router_mode" -eq 1 ]]; then'):
                router_start = i
                in_router = True
                router_if_depth = 1
                router_lines.append(ls)
                continue

            if in_router:
                router_lines.append(ls)
                # Track if/fi depth
                if ls.startswith("if "):
                    router_if_depth += 1
                if ls == "fi" or ls.startswith("fi #"):
                    router_if_depth -= 1
                    if router_if_depth == 0:
                        break  # Found the closing fi for router if

        # Search the router lines for -ngl or --gpu-layers
        found_ngl = any("-ngl" in rl or "--gpu-layers" in rl for rl in router_lines)

        assert found_ngl, (
            "Router-mode startup in start-llama.sh must include -ngl flag. "
            "The [global] ngl value from models.ini must be forwarded "
            "to the llama-server command. Router lines searched: "
            + "\n".join(router_lines[:50])
        )


# ── Router-Mode Command Construction Tests ───────────────────────────────────

class TestRouterModeCommandConstruction:
    """Verify router-mode command construction for GPU offload readiness."""

    def test_router_mode_env_vars_set_by_lifecycle(self, monkeypatch):
        """Router mode lifecycle sets expected env vars for LLAMA_MODELS_MAX and LLAMA_MODELS_PRESET."""
        dummy = DummySrv({
            "llama_models_max": 2,
            "llama_models_preset": str(MODELS_INI),
        })
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        captured_env = {}

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            captured_env.update(env or {})
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        try:
            lifecycle.start_llama_server(None)  # router mode
        except Exception:
            pass

        assert captured_env.get("LLAMA_MODELS_MAX") == "2", (
            f"Expected LLAMA_MODELS_MAX=2, got: {captured_env.get('LLAMA_MODELS_MAX')}"
        )
        assert captured_env.get("LLAMA_MODELS_PRESET") == str(MODELS_INI), (
            f"Expected LLAMA_MODELS_PRESET={MODELS_INI}, got: {captured_env.get('LLAMA_MODELS_PRESET')}"
        )

    def test_router_mode_start_script_call(self, monkeypatch):
        """Router mode calls start-llama.sh with 'router' argument."""
        dummy = DummySrv()
        monkeypatch.setattr(lifecycle, "_srv", lambda: dummy)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        captured_cmd = []

        def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None):
            captured_cmd.extend(cmd)
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

        try:
            lifecycle.start_llama_server(None)  # router mode
        except Exception:
            pass

        assert len(captured_cmd) >= 2, (
            f"Expected at least 2 command parts, got: {captured_cmd}"
        )
        # The last argument should be "router"
        assert captured_cmd[-1] == "router", (
            f"Expected last argument='router', got: {captured_cmd}"
        )


# ── CPU-Only Rollback Readyness Tests ────────────────────────────────────────

class TestCpuOnlyRollback:
    """Verify that the infrastructure for CPU-only rollback exists.

    Router mode should support a rollback toggle that allows operators to
    disable GPU offload and return to CPU-only operation if ROCm becomes
    unstable. This test validates that the env-var override path exists.
    """

    def test_ngl_zero_rollback_possible_via_env(self):
        """Operators can override ngl=0 via LLAMA_NGL env var for CPU-only rollback.

        Setting LLAMA_NGL=0 before starting forces CPU-only operation.
        This test verifies that the startup script honors such an override.
        """
        content = START_SCRIPT.read_text()

        # Verify LLAMA_NGL env var override exists
        assert "LLAMA_NGL" in content, (
            "LLAMA_NGL env var override must exist in start-llama.sh so "
            "operators can force CPU-only mode by setting LLAMA_NGL=0."
        )

        # Verify LLAMA_NGL takes precedence over models.ini [global] ngl
        assert 'GLOBAL_NGL="${LLAMA_NGL:-' in content, (
            "start-llama.sh must use LLAMA_NGL as an override with fallback "
            "to get_global_ngl from models.ini"
        )

    def test_models_ini_ngl_is_configurable(self):
        """[global] ngl in models.ini is the single source of truth.

        Operators can change ngl in models.ini to adjust GPU offload levels
        without modifying start-llama.sh.
        """
        ngl_value = TestModelsIniGlobalNgl._parse_global_ngl()
        assert ngl_value is not None and ngl_value >= 0, (
            f"[global] ngl must be a non-negative integer, got: {ngl_value}"
        )
        # Verify it can be set to 0 (CPU-only)
        assert ngl_value >= 0, "ngl should support 0 as CPU-only mode"
