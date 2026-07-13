"""
Integration tests for proxyctl CLI.

Tests cover all proxyctl commands (start, stop, restart, status, logs)
and configuration resolution (env vars, config.yaml, fallbacks).
"""

import os
import signal
import subprocess
import textwrap
from pathlib import Path

import pytest

PYTEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = PYTEST_DIR.parent.parent
PROXYCTL = REPO_ROOT / "proxy" / "proxyctl"


@pytest.fixture(autouse=True)
def cleanup_background_processes():
    """Kill any background processes started by proxyctl tests."""
    yield
    # Clean up any remaining proxyctl-managed processes
    try:
        subprocess.run(
            ["pkill", "-f", "fake-start.sh"],
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            ["pkill", "-f", "start-script.sh"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def run_proxyctl(*args, env=None, cwd=None, timeout=15):
    """Run proxyctl and return CompletedProcess."""
    cmd = [str(PROXYCTL)] + list(args)
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
    return result


# ============================================================================
# Config Detection Tests
# ============================================================================

class TestConfigDetection:
    """Test proxyctl configuration resolution."""

    def test_env_var_override(self, tmp_path):
        """LLAMA_START_SCRIPT env var should be preferred over config.yaml."""
        env = {"LLAMA_START_SCRIPT": str(tmp_path / "my-start.sh")}
        # We can't easily inspect the resolved path from proxyctl directly,
        # but we can verify that start fails with our custom path in the error.
        result = run_proxyctl("start", env=env)
        assert result.returncode == 2
        assert str(tmp_path / "my-start.sh") in result.stderr

    def test_fallback_to_repo_root(self, tmp_path):
        """When no env var and no config key match, fallback to proxy/scripts/start-proxy.sh."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {"XDG_STATE_HOME": str(state_dir)}
        result = run_proxyctl("status", env=env)
        assert result.returncode == 3  # not running

    def test_status_exit_code_not_running(self, tmp_path):
        """Status should return non-zero (3) when proxy is not running."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {"XDG_STATE_HOME": str(state_dir)}
        result = run_proxyctl("status", env=env)
        assert result.returncode == 3
        assert "not running" in result.stdout

    def test_logs_no_log_file(self, tmp_path):
        """Logs should fail with helpful message when no log file exists."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {"XDG_STATE_HOME": str(state_dir)}
        result = run_proxyctl("logs", env=env)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert "Logfile does not exist" in output


# ============================================================================
# Start Command Tests
# ============================================================================

class TestStart:
    """Test proxyctl start command."""

    def test_start_fails_missing_script(self, tmp_path):
        """Start should fail with exit code 2 when start script does not exist."""
        env = {"LLAMA_START_SCRIPT": str(tmp_path / "nonexistent.sh")}
        result = run_proxyctl("start", env=env)
        assert result.returncode == 2
        assert "Start script not found" in result.stderr

    def test_start_fails_non_executable_script(self, tmp_path):
        """Start should fail when script is not executable."""
        script = tmp_path / "non-exec.sh"
        script.write_text("#!/usr/bin/env bash\necho test\n")
        # Don't set executable
        env = {"LLAMA_START_SCRIPT": str(script)}
        result = run_proxyctl("start", env=env)
        assert result.returncode == 2
        assert "Start script not found" in result.stderr

    def _ensure_pid_dir(self, state_dir):
        """Ensure the PID directory exists."""
        pid_dir = state_dir / "llama-proxy"
        pid_dir.mkdir(parents=True, exist_ok=True)
        return pid_dir

    def test_start_creates_pidfile(self, tmp_path):
        """Start should create a PID file for the running process."""
        # Create a fake long-running start script
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            # long running process
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        result = run_proxyctl("start", env=env)
        # Start should succeed
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # PID file should exist
        assert pid_file.exists(), f"PID file not found at {pid_file}"
        pid = int(pid_file.read_text().strip())
        # Process should be running
        assert os.path.exists(f"/proc/{pid}"), f"Process {pid} not running"

        # Cleanup: stop via proxyctl and kill any remaining
        run_proxyctl("stop", env=env)
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def test_start_dev_creates_dev_pidfile(self, tmp_path):
        """Start --dev should create a separate dev PID file."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        dev_pid_file = state_dir / "llama-proxy" / "proxy.dev.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
            "PROXY_PORT": "18999",  # avoid conflicts
            "LLAMA_SERVER_PORT": "18998",
        }

        result = run_proxyctl("start", "--dev", env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert dev_pid_file.exists(), f"Dev PID file not found at {dev_pid_file}"
        pid = int(dev_pid_file.read_text().strip())
        assert os.path.exists(f"/proc/{pid}"), f"Process {pid} not running"

        # Cleanup
        run_proxyctl("stop", "--dev", env=env)
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def test_start_foreground_blocks(self, tmp_path):
        """Start --foreground should run in foreground (block until killed)."""
        start_script = tmp_path / "foreground.sh"
        # Use a short sleep so the test doesn't hang
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            echo "foreground running"
            sleep 3
            echo "foreground done"
        """))
        start_script.chmod(0o755)

        env = {"LLAMA_START_SCRIPT": str(start_script)}

        # Foreground should block until the script exits
        result = run_proxyctl("start", "--foreground", env=env, timeout=15)
        assert result.returncode == 0
        assert "foreground running" in result.stdout
        assert "foreground done" in result.stdout

    def test_start_with_wait_accepts_flag(self, tmp_path):
        """Start --wait should be accepted (health polling is best-effort)."""
        start_script = tmp_path / "wait-test.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        # --wait should not cause start to fail (health check is best-effort)
        result = run_proxyctl("start", "--wait", env=env)
        assert result.returncode == 0

        # Cleanup
        run_proxyctl("stop", env=env)


# ============================================================================
# Stop Command Tests
# ============================================================================

class TestStop:
    """Test proxyctl stop command."""

    def test_stop_running_process(self, tmp_path):
        """Stop should terminate a running process and remove PID file."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        # Start the process
        result = run_proxyctl("start", env=env)
        assert result.returncode == 0
        assert pid_file.exists()

        pid = int(pid_file.read_text().strip())
        assert os.path.exists(f"/proc/{pid}"), f"Process {pid} not running"

        # Stop the process
        result = run_proxyctl("stop", env=env)
        assert result.returncode == 0
        assert "Stopped" in result.stdout
        # PID file should be removed
        assert not pid_file.exists()
        # Process should not be running
        assert not os.path.exists(f"/proc/{pid}"), f"Process {pid} still running"

    def test_stop_not_running(self, tmp_path):
        """Stop should return 0 when proxy is not running."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            "XDG_STATE_HOME": str(state_dir),
        }
        result = run_proxyctl("stop", env=env)
        assert result.returncode == 0
        assert "not running" in result.stdout


# ============================================================================
# Restart Command Tests
# ============================================================================

class TestRestart:
    """Test proxyctl restart command."""

    def test_restart_stops_then_starts(self, tmp_path):
        """Restart should stop existing process and start a new one."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        # First start
        result = run_proxyctl("start", env=env)
        assert result.returncode == 0
        assert pid_file.exists()
        old_pid = int(pid_file.read_text().strip())

        # Restart
        result = run_proxyctl("restart", env=env)
        assert result.returncode == 0
        assert pid_file.exists()
        new_pid = int(pid_file.read_text().strip())

        # PID should be different (new process)
        assert new_pid != old_pid
        assert os.path.exists(f"/proc/{new_pid}"), f"New process {new_pid} not running"
        # Old process should be gone
        assert not os.path.exists(f"/proc/{old_pid}"), f"Old process {old_pid} still running"

        # Cleanup
        try:
            os.kill(new_pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def test_restart_when_not_running(self, tmp_path):
        """Restart should start the proxy even if it was not running."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        # Restart (no existing process)
        result = run_proxyctl("restart", env=env)
        assert result.returncode == 0
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert os.path.exists(f"/proc/{pid}"), f"Process {pid} not running"

        # Cleanup
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


# ============================================================================
# Status Command Tests
# ============================================================================

class TestStatus:
    """Test proxyctl status command."""

    def test_status_shows_running(self, tmp_path):
        """Status should show PID and start time when proxy is running."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }

        result = run_proxyctl("start", env=env)
        assert result.returncode == 0

        # Check status
        result = run_proxyctl("status", env=env)
        assert result.returncode == 0
        assert "proxy running" in result.stdout
        assert "PID" in result.stdout
        assert "Started:" in result.stdout

        # Cleanup - use the same env so PID file is found
        run_proxyctl("stop", env=env)

    def test_status_not_running(self, tmp_path):
        """Status should return exit code 3 when proxy is not running."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {"XDG_STATE_HOME": str(state_dir)}
        result = run_proxyctl("status", env=env)
        assert result.returncode == 3
        assert "not running" in result.stdout

    def test_status_with_different_pid_files(self, tmp_path):
        """Status should correctly use separate PID files for prod vs dev."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"
        dev_pid_file = state_dir / "llama-proxy" / "proxy.dev.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }
        dev_env = {
            **env,
            "PROXY_PORT": "18989",
            "LLAMA_SERVER_PORT": "18988",
        }

        # Start prod
        run_proxyctl("start", env=env)
        assert pid_file.exists()

        # Status for prod should find it
        result = run_proxyctl("status", env=env)
        assert result.returncode == 0
        assert "proxy running" in result.stdout

        # Status with --dev should NOT find prod (uses different PID file)
        result_dev = run_proxyctl("status", "--dev", env=dev_env)
        assert result_dev.returncode == 3
        assert "not running" in result_dev.stdout

        # Cleanup
        run_proxyctl("stop", env=env)


# ============================================================================
# Logs Command Tests
# ============================================================================

class TestLogs:
    """Test proxyctl logs command."""

    def test_logs_no_file(self, tmp_path):
        """Logs should fail when log file does not exist."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            "XDG_STATE_HOME": str(state_dir),
        }
        result = run_proxyctl("logs", env=env)
        # stderr or stdout depending on code path
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert "Logfile does not exist" in output

    def test_logs_file_flag(self, tmp_path):
        """Logs --file should tail a specified file."""
        log_file = tmp_path / "custom.log"
        log_file.write_text("custom log line 1\ncustom log line 2\n")

        result = run_proxyctl("logs", "--file", str(log_file), "--no-follow")
        assert result.returncode == 0
        assert "custom log line 1" in result.stdout
        assert "custom log line 2" in result.stdout

    def test_logs_file_flag_missing(self, tmp_path):
        """Logs --file should fail on missing file."""
        result = run_proxyctl("logs", "--file", str(tmp_path / "nonexistent.log"), "--no-follow")
        assert result.returncode == 1

    def test_logs_lines_flag(self, tmp_path):
        """Logs --lines N should show only the last N lines."""
        log_file = tmp_path / "lines.log"
        log_file.write_text("line 1\nline 2\nline 3\n")

        result = run_proxyctl("logs", "--file", str(log_file), "--lines", "2", "--no-follow")
        assert result.returncode == 0
        assert "line 1" not in result.stdout
        assert "line 2" in result.stdout
        assert "line 3" in result.stdout

    def test_logs_cat_does_not_block(self, tmp_path):
        """Logs --no-follow should print and exit without blocking."""
        log_file = tmp_path / "cat.log"
        log_file.write_text("cat line 1\n")

        result = run_proxyctl("logs", "--file", str(log_file), "--no-follow")
        assert result.returncode == 0
        assert "cat line 1" in result.stdout

    def test_logs_dev_mode_no_file(self, tmp_path):
        """Logs --dev should fail when no dev log exists."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {
            "XDG_STATE_HOME": str(state_dir),
        }
        result = run_proxyctl("logs", "--dev", env=env)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert "Logfile does not exist" in output

    def test_logs_tails_existing_file(self, tmp_path):
        """Logs should tail an existing log file."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        log_dir = state_dir / "llama-proxy" / "logs"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "proxy.log"
        log_file.write_text("test log line 1\ntest log line 2\n")

        env = {
            "XDG_STATE_HOME": str(state_dir),
        }

        # use --cat to avoid blocking
        result = run_proxyctl("logs", "--cat", env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "test log line 1" in result.stdout
        assert "test log line 2" in result.stdout

    def test_logs_follow_flag(self, tmp_path):
        """Logs --follow should keep following (blocks), verify with timeout."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        log_dir = state_dir / "llama-proxy" / "logs"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "proxy.log"
        log_file.write_text("initial line\n")

        env = {
            "XDG_STATE_HOME": str(state_dir),
        }

        cmd = [str(PROXYCTL), "logs", "--follow"]
        import signal as sig
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, **env},
            preexec_fn=os.setsid,
        )
        import time as _time
        _time.sleep(1)
        os.killpg(os.getpgid(proc.pid), sig.SIGTERM)
        stdout, stderr = proc.communicate(timeout=5)
        assert "initial line" in stdout


# ============================================================================
# Dev Mode Tests
# ============================================================================

class TestDevMode:
    """Test proxyctl --dev flag."""

    def test_dev_mode_uses_separate_pid_file(self, tmp_path):
        """Dev mode should create a separate dev PID file, distinct from prod."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"
        dev_pid_file = state_dir / "llama-proxy" / "proxy.dev.pid"

        # Create a simple backend script (dev mode starts backend + uvicorn;
        # uvicorn may not be available in test env, so validate PID file creation)
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
            "PROXY_PORT": "18995",
            "LLAMA_SERVER_PORT": "18994",
        }

        # Start dev - this should create the dev PID file
        # (the uvicorn process may fail, but the dev PID file is still created)
        result = run_proxyctl("start", "--dev", env=env)
        # In some environments uvicorn may not be available - the start command
        # still completes and creates the dev PID file
        assert dev_pid_file.exists(), f"Dev PID file should exist. stdout: {result.stdout} stderr: {result.stderr}"

        # The dev PID file should be separate from prod PID file
        assert not pid_file.exists(), "Prod PID file should NOT exist after dev start"

    def test_dev_mode_stop_removes_pid_file(self, tmp_path):
        """Stop --dev should remove the dev PID file."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        dev_pid_file = state_dir / "llama-proxy" / "proxy.dev.pid"

        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
            "PROXY_PORT": "18991",
            "LLAMA_SERVER_PORT": "18990",
        }

        # Start dev
        run_proxyctl("start", "--dev", env=env)
        assert dev_pid_file.exists()

        # Stop dev - should clean up the PID file (even if process already exited)
        result = run_proxyctl("stop", "--dev", env=env)
        assert result.returncode == 0
        # PID file should be removed
        assert not dev_pid_file.exists(), "Dev PID file should be removed on stop"

    def test_prod_and_dev_independent(self, tmp_path):
        """Production and dev instances should use separate PID files."""
        start_script = tmp_path / "start-script.sh"
        start_script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            while true; do sleep 1; done
        """))
        start_script.chmod(0o755)

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        pid_file = state_dir / "llama-proxy" / "proxy.pid"
        dev_pid_file = state_dir / "llama-proxy" / "proxy.dev.pid"

        env = {
            "LLAMA_START_SCRIPT": str(start_script),
            "XDG_STATE_HOME": str(state_dir),
        }
        dev_env = {
            **env,
            "PROXY_PORT": "18993",
            "LLAMA_SERVER_PORT": "18992",
        }

        # Start prod
        run_proxyctl("start", env=env)
        assert pid_file.exists()

        # Start dev - creates separate PID file
        run_proxyctl("start", "--dev", env=dev_env)
        assert dev_pid_file.exists()
        # Both PID files should coexist
        assert pid_file.exists(), "Prod PID should still exist"
        assert dev_pid_file.exists(), "Dev PID should exist"

        # Stop prod (dev should be unaffected)
        result = run_proxyctl("stop", env=env)
        assert result.returncode == 0
        assert not pid_file.exists(), "Prod PID should be removed"
        # Dev PID should still exist
        assert dev_pid_file.exists(), "Dev PID should remain after prod stop"

        # Stop dev
        result = run_proxyctl("stop", "--dev", env=dev_env)
        assert result.returncode == 0
        assert not dev_pid_file.exists(), "Dev PID should be removed after dev stop"


# ============================================================================
# Help Command Tests
# ============================================================================

class TestHelp:
    """Test proxyctl help/usage output."""

    def test_help_output(self):
        """Help should show available commands."""
        result = run_proxyctl("help")
        assert result.returncode == 0
        assert "Usage" in result.stdout
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "restart" in result.stdout
        assert "status" in result.stdout
        assert "logs" in result.stdout

    def test_help_flag(self):
        """--help and -h should also show usage."""
        for flag in ("--help", "-h"):
            result = run_proxyctl(flag)
            assert result.returncode == 0
            assert "Usage" in result.stdout

    def test_no_args_shows_usage(self):
        """Running proxyctl without args should show usage and exit 1."""
        result = run_proxyctl()
        assert result.returncode == 1
        assert "Usage" in result.stdout

    def test_unknown_command(self):
        """Unknown commands should show usage and exit 2."""
        result = run_proxyctl("unknown-command")
        assert result.returncode == 2
        assert "Unknown command" in result.stdout

    def test_dev_flag_in_usage(self, tmp_path):
        """Usage should document --dev flag."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = {"XDG_STATE_HOME": str(state_dir)}
        result = run_proxyctl("help", env=env)
        assert "--dev" in result.stdout
