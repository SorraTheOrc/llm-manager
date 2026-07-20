"""
Integration tests for proxyctl config detection and path resolution.

Uses the debug-resolve command to assert resolved paths without side effects
(starting/stopping real proxy processes). Tests cover all 5 acceptance criteria.
"""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

PYTEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = PYTEST_DIR.parent.parent
PROXYCTL = REPO_ROOT / "proxy" / "proxyctl"


def run_debug_resolve(*, proxyctl_path=None, env=None, cwd=None, timeout=10):
    """Run proxyctl debug-resolve and return dict of key=value pairs."""
    cmd = [str(proxyctl_path or PROXYCTL), "debug-resolve"]
    proc_env = os.environ.copy()
    # Unset XDG_RUNTIME_DIR by default so PID_DIR falls back to XDG_STATE_HOME.
    # Tests that want XDG_RUNTIME_DIR behavior should set it explicitly in env.
    proc_env.pop("XDG_RUNTIME_DIR", None)
    if env:
        proc_env.update(env)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or str(REPO_ROOT),
        env=proc_env,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"debug-resolve failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    )
    # Parse key=value lines
    parsed = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed


def copy_proxyctl_and_make_fake_proxy(tmp_path):
    """Copy proxyctl to a temporary proxy/ dir and return the path to the copy."""
    tmp_proxy = tmp_path / "proxy"
    tmp_proxy.mkdir()
    dst = tmp_proxy / "proxyctl"
    # Copy executable preserving permissions
    shutil.copy2(str(PROXYCTL), str(dst))
    dst.chmod(0o755)
    return dst, tmp_proxy


# ============================================================================
# AC1: LLAMA_START_SCRIPT env var override
# ============================================================================


class TestAC1_EnvVarOverride:
    """AC1: When LLAMA_START_SCRIPT env var is set, proxyctl uses it."""

    def test_env_var_override(self, tmp_path):
        """LLAMA_START_SCRIPT should be preferred over all other resolution."""
        my_script = str(tmp_path / "from-env.sh")
        env = {
            "LLAMA_START_SCRIPT": my_script,
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(env=env)
        assert result["start_script"] == my_script, (
            f"Expected start_script={my_script}, got {result['start_script']}"
        )


# ============================================================================
# AC2: config.yaml server.llama_start_script override
# ============================================================================


class TestAC2_ConfigOverride:
    """AC2: When config.yaml contains llama_start_script, proxyctl uses it."""

    def test_config_start_script_absolute(self, tmp_path):
        """Absolute path in config.yaml is used as-is."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent("""\
            server:
              llama_start_script: /tmp/from-config-abs.sh
        """))
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        assert result["start_script"] == "/tmp/from-config-abs.sh", (
            f"Expected /tmp/from-config-abs.sh, got {result['start_script']}"
        )

    def test_config_start_script_relative(self, tmp_path):
        """Relative path in config.yaml is resolved against repo root."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent("""\
            server:
              llama_start_script: scripts/start-proxy.sh
        """))
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        # Relative path should resolve against the repo root (parent of proxy/)
        expected = str(tmp_proxy.parent / "scripts" / "start-proxy.sh")
        assert result["start_script"] == expected, (
            f"Expected {expected}, got {result['start_script']}"
        )

    def test_env_var_takes_precedence_over_config(self, tmp_path):
        """LLAMA_START_SCRIPT env var should take precedence over config.yaml."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent("""\
            server:
              llama_start_script: /tmp/from-config.sh
        """))
        env = {
            "LLAMA_START_SCRIPT": "/tmp/from-env.sh",
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        assert result["start_script"] == "/tmp/from-env.sh", (
            f"Expected env override, got {result['start_script']}"
        )


# ============================================================================
# AC3: Default fallback to proxy/scripts/start-proxy.sh
# ============================================================================


class TestAC3_DefaultFallback:
    """AC3: With no overrides, proxyctl resolves proxy/scripts/start-proxy.sh."""

    def test_default_fallback(self, tmp_path):
        """With no env var and no config key, fallback to proxy/scripts/start-proxy.sh."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        # Create scripts/start-proxy.sh so the path resolves
        scripts = tmp_proxy / "scripts"
        scripts.mkdir()
        start_proxy = scripts / "start-proxy.sh"
        start_proxy.write_text("#!/usr/bin/env bash\necho test\n")
        start_proxy.chmod(0o755)

        # No LLAMA_START_SCRIPT, no config.yaml
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        expected = str(start_proxy.resolve())
        assert result["start_script"] == expected, (
            f"Expected fallback {expected}, got {result['start_script']}"
        )

    def test_no_config_file_fallback(self, tmp_path):
        """Without config.yaml, fallback to proxy/scripts/start-proxy.sh."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        # Create scripts/start-proxy.sh (no config.yaml)
        scripts = tmp_proxy / "scripts"
        scripts.mkdir()
        start_proxy = scripts / "start-proxy.sh"
        start_proxy.write_text("#!/usr/bin/env bash\necho test\n")
        start_proxy.chmod(0o755)

        # Clean env: no LLAMA_START_SCRIPT, no XDG_RUNTIME_DIR
        env = {"XDG_STATE_HOME": str(tmp_path / "state")}
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        expected = str(start_proxy.resolve())
        assert result["start_script"] == expected, (
            f"Expected fallback {expected}, got {result['start_script']}"
        )


# ============================================================================
# AC4: XDG_RUNTIME_DIR for pidfile, fallback to XDG_STATE_HOME
# ============================================================================


class TestAC4_PidFilePath:
    """AC4: Default pidfile uses XDG_RUNTIME_DIR when set, falls back to XDG_STATE_HOME."""

    def test_xdg_runtime_dir_used_for_pid_dir(self, tmp_path):
        """When XDG_RUNTIME_DIR is set, PID_DIR should be under it."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        env = {
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(env=env)
        expected_pid_dir = str(runtime_dir / "llama-proxy")
        assert result["pid_dir"] == expected_pid_dir, (
            f"Expected pid_dir={expected_pid_dir}, got {result['pid_dir']}"
        )
        expected_pid_file = str(runtime_dir / "llama-proxy" / "proxy.pid")
        assert result["pid_file"] == expected_pid_file, (
            f"Expected pid_file={expected_pid_file}, got {result['pid_file']}"
        )

    def test_xdg_state_home_fallback_for_pid_dir(self, tmp_path):
        """When XDG_RUNTIME_DIR is not set, PID_DIR should use XDG_STATE_HOME."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            # XDG_RUNTIME_DIR not set (run_debug_resolve pops it)
            "XDG_STATE_HOME": str(state_dir),
        }
        result = run_debug_resolve(env=env)
        expected_pid_dir = str(state_dir / "llama-proxy")
        assert result["pid_dir"] == expected_pid_dir, (
            f"Expected pid_dir={expected_pid_dir}, got {result['pid_dir']}"
        )

    def test_xdg_runtime_dir_dev_pid_file(self, tmp_path):
        """Dev pid file should also be under XDG_RUNTIME_DIR when set."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        env = {
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(env=env)
        expected_dev = str(runtime_dir / "llama-proxy" / "proxy.dev.pid")
        assert result["dev_pid_file"] == expected_dev, (
            f"Expected dev_pid_file={expected_dev}, got {result['dev_pid_file']}"
        )


# ============================================================================
# AC5: Logging directory from config.yaml with writability fallback
# ============================================================================


class TestAC5_LogDir:
    """AC5: Logging directory from config.yaml with permission-fallback."""

    def test_config_log_dir_is_resolved(self, tmp_path):
        """resolve_log_dir should use logging.directory from config.yaml."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        log_dir = tmp_path / "my-logs"
        log_dir.mkdir()
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent(f"""\
            logging:
              directory: {log_dir}
        """))
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        assert result["resolved_log_dir"] == str(log_dir), (
            f"Expected resolved_log_dir={log_dir}, got {result['resolved_log_dir']}"
        )

    def test_config_log_dir_not_writable_falls_back_to_repo_local(self, tmp_path):
        """When configured log dir is not writable, fallback to repo-local proxy/logs/."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        # Point to /var/log which is not writable by non-root
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent("""\
            logging:
              directory: /var/log/llama-proxy-test-do-not-use
        """))
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        expected_fallback = str(tmp_proxy / "logs")
        assert result["resolved_log_dir"] == expected_fallback, (
            f"Expected fallback to repo-local {expected_fallback}, "
            f"got {result['resolved_log_dir']}"
        )

    def test_no_config_log_dir_uses_default(self, tmp_path):
        """With no logging.directory in config, resolve_log_dir returns default LOG_DIR."""
        dst, tmp_proxy = copy_proxyctl_and_make_fake_proxy(tmp_path)
        # Create a config.yaml WITHOUT logging.directory
        config = tmp_proxy / "config.yaml"
        config.write_text(textwrap.dedent("""\
            server:
              port: 8080
        """))
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(proxyctl_path=dst, cwd=tmp_path, env=env)
        # Default LOG_DIR is $XDG_STATE_HOME/llama-proxy/logs
        expected = str(tmp_path / "state" / "llama-proxy" / "logs")
        assert result["resolved_log_dir"] == expected, (
            f"Expected default {expected}, got {result['resolved_log_dir']}"
        )


# ============================================================================
# Cross-cutting: debug-resolve basic operation
# ============================================================================


class TestDebugResolve:
    """Verify debug-resolve command itself works correctly."""

    def test_prints_all_expected_keys(self, tmp_path):
        """debug-resolve should print all expected keys."""
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(env=env)
        expected_keys = {
            "start_script", "pid_file", "log_file", "config_file",
            "pid_dir", "log_dir", "resolved_log_dir",
            "dev_pid_file", "dev_log_file",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_config_file_is_absolute(self, tmp_path):
        """config_file should be an absolute path."""
        env = {
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
        result = run_debug_resolve(env=env)
        assert Path(result["config_file"]).is_absolute(), (
            f"config_file should be absolute: {result['config_file']}"
        )


# ============================================================================
# Helper test start script
# ============================================================================


class TestHelperStartScript:
    """Verify the default start script (as resolved from config.yaml) works."""

    def test_start_script_is_found(self, tmp_path):
        """The resolved start script (from config.yaml or fallback) should exist and be executable."""
        env = {"XDG_STATE_HOME": str(tmp_path / "state")}
        result = run_debug_resolve(env=env)
        start_script = result["start_script"]
        # The resolved path should exist
        assert Path(start_script).exists(), (
            f"Start script not found: {start_script}"
        )
        # The file should be executable
        assert os.access(start_script, os.X_OK), (
            f"Start script not executable: {start_script}"
        )


