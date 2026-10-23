#!/usr/bin/env python3
"""Behavioural tests for ``start-proxy.sh`` launch modes.

Pins the flag contract of LP-0MVK3KIPZ002SR32 (parent LP-0MVGT0E6W000Q8CY):

* the default launch detaches (own session via ``setsid``, stdio to the log
  file) and prints a single handoff line, so a launcher killed by an agent
  tool-call timeout cannot take the server down;
* ``--foreground`` and ``START_PROXY_FOREGROUND=1`` restore today's blocking
  ``exec`` (AC2), and the flag is consumed — never forwarded to uvicorn;
* the detached re-execution drops ``--restart`` so the kill/cleanup phase runs
  exactly once (no repeated restart storm);
* an unwritable ``--log-file`` fails fast instead of detaching into silence;
* the usage header documents both flags and the env opt-in.

The pgid-SIGKILL survival regression itself lives in the shell harness
``tests/test_start_proxy_detach.sh`` (wrapped by
``tests/test_start_proxy_detach_shell.py``); this module covers the flag
surface that the harness does not.

Sandbox: a copy of the script, a fake ``python3`` that intercepts the final
``-m uvicorn`` (recording argv and exiting, so no real server ever starts), an
isolated ``$HOME``, an ephemeral port, and stubbed ``ss``/``netstat``
port probes plus stubbed ``pkill``/``fuser`` — so the live proxy, ports
8000/8080/8081 and any real process are never touched.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = REPO_ROOT / "proxy" / "scripts" / "start-proxy.sh"
REAL_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable

FAKE_PY = """#!/usr/bin/env bash
# Intercept only the final `exec <python> -m uvicorn ...`; delegate the script's
# own helper calls (merged-config hashing, key resolution) to the real
# interpreter.
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    printf '%s\\n' "$*" >> "$FAKE_UVICORN_ARGV"
    echo "FAKE_UVICORN $*"
    exit 0
fi
exec "@REAL_PY@" "$@"
"""


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _sandbox(tmp_path: Path) -> dict[str, str]:
    """Build an isolated proxy tree and return the environment for a run."""
    proxy_dir = tmp_path / "proxy"
    (proxy_dir / "scripts").mkdir(parents=True)
    (proxy_dir / "proxy").mkdir()
    shutil.copy2(SOURCE_SCRIPT, proxy_dir / "scripts" / "start-proxy.sh")
    for helper in ("config_merge.py", "api_key_resolver.py"):
        src = REPO_ROOT / "proxy" / "proxy" / helper
        shutil.copy2(src, proxy_dir / "proxy" / helper)

    # No api_key_env: key resolution is a no-op, so no provider key is needed.
    (proxy_dir / "config.yaml").write_text(
        "models:\n"
        "  test-model:\n"
        "    providers:\n"
        "      - name: test-local\n"
        "        type: local\n"
        "        llama_model: Test\n"
        "server:\n"
        "  session_slot_pool_size: 1\n",
        encoding="utf-8",
    )

    home = tmp_path / "home"
    (home / ".pi" / "agent").mkdir(parents=True)
    (home / ".pi" / "agent" / "auth.json").write_text("{}\n", encoding="utf-8")

    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").write_text(FAKE_PY.replace("@REAL_PY@", REAL_PY), encoding="utf-8")
    (tools / "python3").chmod(0o755)
    # Port probes report "free"; kill helpers are stubs so a --restart run can
    # never reach a live pkill/fuser (project rule: never blanket-kill python).
    for name in ("ss", "netstat", "pkill", "fuser"):
        (tools / name).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        (tools / name).chmod(0o755)

    argv_log = tmp_path / "uvicorn-argv.log"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "START_PROXY_FOREGROUND",
            "LLAMA_PROXY_USE_ENV_KEYS",
            "PROXY_PORT",
            "PORT",
            "PYTHONPATH",
        )
    }
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": f"{proxy_dir}{os.pathsep}{proxy_dir / 'proxy'}",
            "FAKE_UVICORN_ARGV": str(argv_log),
        }
    )
    return {"proxy_dir": str(proxy_dir), "env": env, "argv_log": argv_log}


def _run(sandbox: dict, *args: str, extra_env: dict[str, str] | None = None):
    env = dict(sandbox["env"])
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", "scripts/start-proxy.sh", *args],
        cwd=sandbox["proxy_dir"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _uvicorn_calls(argv_log: Path, expected: int = 1, timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while True:
        calls = (
            [line for line in argv_log.read_text(encoding="utf-8").splitlines() if line]
            if argv_log.exists()
            else []
        )
        if len(calls) >= expected or time.monotonic() > deadline:
            return calls
        time.sleep(0.05)


def test_foreground_flag_keeps_blocking_exec_and_is_not_forwarded(tmp_path):
    """``--foreground`` runs uvicorn in-process and never reaches uvicorn."""
    sandbox = _sandbox(tmp_path)
    port = _free_port()

    proc = _run(sandbox, "--foreground", "--log-file", "/tmp/unused-detach.log", "--port", str(port))

    assert proc.returncode == 0, proc.stderr
    calls = _uvicorn_calls(sandbox["argv_log"])
    assert len(calls) == 1, f"expected exactly one foreground uvicorn launch, got {calls}"
    assert "--foreground" not in calls[0]
    assert "--log-file" not in calls[0]
    assert f"--port {port}" in calls[0]
    assert "Proxy launched detached" not in proc.stdout


def test_start_proxy_foreground_env_opt_in_keeps_blocking_exec(tmp_path):
    """``START_PROXY_FOREGROUND=1`` is the unattended opt-in for the same path."""
    sandbox = _sandbox(tmp_path)
    port = _free_port()

    proc = _run(
        sandbox,
        "--port",
        str(port),
        extra_env={"START_PROXY_FOREGROUND": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    calls = _uvicorn_calls(sandbox["argv_log"])
    assert len(calls) == 1, f"expected exactly one foreground uvicorn launch, got {calls}"
    assert "Proxy launched detached" not in proc.stdout


def test_default_launch_detaches_and_prints_one_handoff_line(tmp_path):
    """By default the launcher hands off and exits instead of serving."""
    sandbox = _sandbox(tmp_path)
    port = _free_port()
    log_file = tmp_path / "detached.log"

    proc = _run(sandbox, "--port", str(port), "--log-file", str(log_file))

    assert proc.returncode == 0, proc.stderr
    handoff = [line for line in proc.stdout.splitlines() if str(log_file) in line]
    assert len(handoff) == 1, f"expected exactly one handoff line, got {handoff}"
    assert f"curl -sf http://127.0.0.1:{port}/health" in handoff[0]
    assert "FAKE_UVICORN" not in proc.stdout, "the child's output leaked into the caller's stdout"

    # The detached child re-executes the script with --foreground and reaches
    # the uvicorn exec exactly once.
    calls = _uvicorn_calls(sandbox["argv_log"])
    assert len(calls) == 1, f"expected exactly one detached uvicorn launch, got {calls}"
    assert f"--port {port}" in calls[0]


def test_detached_reexec_does_not_repeat_the_restart_phase(tmp_path):
    """The child must not re-enter ``--restart`` (no repeated restart storm)."""
    sandbox = _sandbox(tmp_path)
    port = _free_port()
    log_file = tmp_path / "detached.log"

    proc = _run(sandbox, "--restart", "--port", str(port), "--log-file", str(log_file))

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("Restart requested:") == 1, proc.stdout
    calls = _uvicorn_calls(sandbox["argv_log"])
    assert len(calls) == 1, f"expected exactly one uvicorn launch, got {calls}"

    # The child's own output lands in the log file; a re-entered restart phase
    # would show its banner there a second time.
    deadline = time.monotonic() + 10
    log_text = ""
    while time.monotonic() < deadline:
        log_text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        if "FAKE_UVICORN" in log_text:
            break
        time.sleep(0.05)
    assert "Restart requested:" not in log_text


def test_nohup_backgrounded_caller_pattern_launches_once_and_returns(tmp_path):
    """LP-0MVGT0E6W000Q8CY AC6: the pre-existing detached caller patterns
    (``nohup bash scripts/start-proxy.sh --restart &>/tmp/proxy-startup.log &``
    in the start-proxy skill and ``scripts/rebuild-and-restart-mtp.sh``) must
    keep working with the new self-detaching default: the caller still gets a
    working server, but the caller's own background job now exits promptly
    (the script hands off instead of serving), and the server is launched
    exactly once with one handoff line. No double-detach breakage."""
    sandbox = _sandbox(tmp_path)
    caller_log = tmp_path / "caller.log"

    with caller_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            ["nohup", "bash", "scripts/start-proxy.sh", "--restart"],
            cwd=sandbox["proxy_dir"],
            env=sandbox["env"],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )

    # The caller's job must exit by itself after the handoff (it no longer
    # blocks serving the proxy).
    assert proc.wait(timeout=60) == 0
    calls = _uvicorn_calls(sandbox["argv_log"])
    assert len(calls) == 1, f"expected exactly one uvicorn launch, got {calls}"

    log_text = caller_log.read_text(encoding="utf-8")
    assert log_text.count("Proxy launched detached:") == 1, log_text
    assert "Proxy launched detached:" in log_text
    assert "--restart" in log_text or "Restart" in log_text


def test_unwritable_log_file_fails_fast(tmp_path):
    """A detached launch whose log cannot be written is refused, not silent."""
    sandbox = _sandbox(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n", encoding="utf-8")

    proc = _run(
        sandbox,
        "--port",
        str(_free_port()),
        "--log-file",
        str(blocker / "detached.log"),
    )

    assert proc.returncode != 0
    assert "--log-file" in proc.stderr
    assert "Proxy launched detached" not in proc.stdout
    assert not sandbox["argv_log"].exists(), "the server was launched despite an unusable log"


def test_usage_header_documents_foreground_and_log_file():
    header = SOURCE_SCRIPT.read_text(encoding="utf-8").splitlines()[:40]
    joined = "\n".join(header)
    assert "--foreground" in joined
    assert "--log-file" in joined
    assert "START_PROXY_FOREGROUND" in joined
