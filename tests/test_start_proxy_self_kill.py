#!/usr/bin/env python3
"""Regression test: the launcher must never SIGTERM itself (D2).

LP-0MVZ810NG008A3EW D2. ``start-proxy.sh --restart`` used ``pkill -f
'<pattern>'``, which matches the **caller's own argv**. A restart invoked as

    ./start-proxy.sh --restart --reason "… forced llama-server restart"

therefore SIGTERMed the launcher itself (exit 143) during phase 1, leaving the
proxy stopped and llama-server dying without the script ever reaching its start
phase. The fix selects victims with ``pgrep -f`` (which never matches itself)
and then filters out this process, its parent and any copy of the launcher
before signalling with ``kill``.

What this module pins, against a sandboxed copy of the real script:

1. a pattern-matching process that is **not** the launcher is really killed
   (the helper still does its job);
2. the launcher survives a restart whose own ``--reason`` text contains the
   pattern, and still reaches the launch phase (exit 0 + fake uvicorn argv);
3. the surviving launcher does not depend on the command-line exclusion: the
   script is invoked through a ``launcher.sh`` symlink, so its own ``cmdline``
   no longer contains ``start-proxy.sh`` and only the pid checks can protect it.

Test doubles, and why: ``pgrep`` is stubbed to report the launcher's own pid
(``$PPID``, i.e. exactly what a real ``pgrep -f`` returns when ``--reason``
carries the pattern) plus the pid of a real dummy ``sleep`` process. ``kill`` is
the real binary, so the dummy's death is a genuine signal — but the only
processes that can ever be signalled by phases 1-2 are the two pids the stub
names.

The **port fallback (phase 3) cannot be sandboxed by a ``pgrep`` stub**: it kills
whatever holds ports 8080/8081. LP-0MW3PLEH8008O4I7 measured that leak on the
live host — every run of this module sent ``SIGKILL`` to the live ``tts-server``
(and the live ``llama-server`` router), which is exactly the class-A backend
death the attribution work (docs/death-attribution-LP-0MW3K7WYV001XLL1.md) was
hunting. The sandbox therefore sets ``START_PROXY_PORT_KILL_DISABLED=1`` and
stubs ``fuser``; one test proves the fallback still runs when the guard is not
set. The live proxy, llama-server, TTS server and ports 8000/8080/8081 are never
touched.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_ROOT = REPO_ROOT / "proxy"
PROXY_PKG_DIR = PROXY_ROOT / "proxy"
SOURCE_SCRIPT = PROXY_ROOT / "scripts" / "start-proxy.sh"
REAL_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable

CONFIG_YAML = (
    "models:\n"
    "  test-model:\n"
    "    providers:\n"
    "      - name: test-local\n"
    "        type: local\n"
    "        llama_model: Test\n"
    "server:\n"
    "  session_slot_pool_size: 1\n"
    "  llama_server_port: 8080\n"
)

FAKE_PY = """#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    printf '%s\\n' "$*" >> "$FAKE_UVICORN_ARGV"
    echo "FAKE_UVICORN $*"
    exit 0
fi
exec "@REAL_PY@" "$@"
"""

#: Records the port-kill fallback (phase 3) instead of signalling whatever holds
#: the port. LP-0MW3PLEH8008O4I7: with a real ``fuser`` this module SIGKILLed the
#: live tts-server and llama-server router on every run.
FAKE_FUSER = """#!/usr/bin/env bash
echo "$*" >> "$FUSER_LOG"
exit 0
"""

#: Report the launcher's own pid (what a real `pgrep -f` returns for a --reason
#: containing the pattern) and the pid of the real dummy victim.
FAKE_PGREP = """#!/usr/bin/env bash
printf '%s\\n' "$PPID"
cat "$PGREP_DUMMY_PID_FILE" 2>/dev/null || true
exit 0
"""


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@dataclass
class _Sandbox:
    proxy_dir: Path
    env: dict[str, str]
    argv_log: Path
    launcher: Path
    fuser_log: Path


def _sandbox(tmp_path: Path, *, via_symlink: bool, port_kill_disabled: bool = True) -> _Sandbox:
    proxy_dir = tmp_path / "proxy"
    (proxy_dir / "scripts").mkdir(parents=True)
    (proxy_dir / "proxy").mkdir()
    script = proxy_dir / "scripts" / "start-proxy.sh"
    shutil.copy2(SOURCE_SCRIPT, script)
    for helper in ("config_merge.py", "api_key_resolver.py"):
        shutil.copy2(PROXY_PKG_DIR / helper, proxy_dir / "proxy" / helper)
    (proxy_dir / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    # Invoking through a differently-named symlink removes "start-proxy.sh" from
    # the launcher's own cmdline, so only the pid-based guards can save it.
    launcher = script
    if via_symlink:
        launcher = proxy_dir / "scripts" / "launcher.sh"
        launcher.symlink_to(script)

    home = tmp_path / "home"
    (home / ".pi" / "agent").mkdir(parents=True)
    (home / ".pi" / "agent" / "auth.json").write_text("{}\n", encoding="utf-8")

    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").write_text(FAKE_PY.replace("@REAL_PY@", REAL_PY), encoding="utf-8")
    (tools / "python3").chmod(0o755)
    for name in ("ss", "netstat"):
        (tools / name).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        (tools / name).chmod(0o755)
    (tools / "pgrep").write_text(FAKE_PGREP, encoding="utf-8")
    (tools / "pgrep").chmod(0o755)
    (tools / "fuser").write_text(FAKE_FUSER, encoding="utf-8")
    (tools / "fuser").chmod(0o755)
    # No `kill` stub on purpose: the real signal must reach the dummy.

    dummy_pid_file = tmp_path / "dummy.pid"
    argv_log = tmp_path / "uvicorn-argv.log"
    fuser_log = tmp_path / "fuser.log"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "START_PROXY_FOREGROUND",
            "START_PROXY_PORT_KILL_DISABLED",
            "LLAMA_PROXY_USE_ENV_KEYS",
            "LLAMA_PROXY_RESTART_REASON",
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
            "PGREP_DUMMY_PID_FILE": str(dummy_pid_file),
            "FUSER_LOG": str(fuser_log),
        }
    )
    if port_kill_disabled:
        env["START_PROXY_PORT_KILL_DISABLED"] = "1"
    return _Sandbox(
        proxy_dir=proxy_dir,
        env=env,
        argv_log=argv_log,
        launcher=launcher,
        fuser_log=fuser_log,
    )


def _run(sandbox: _Sandbox, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(sandbox.launcher), *args],
        cwd=str(sandbox.proxy_dir),
        env=sandbox.env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _uvicorn_argv(sandbox: _Sandbox) -> list[str]:
    deadline = time.monotonic() + 10
    while True:
        if sandbox.argv_log.exists():
            lines = [ln for ln in sandbox.argv_log.read_text(encoding="utf-8").splitlines() if ln]
            if lines:
                return lines
        if time.monotonic() > deadline:
            return []
        time.sleep(0.05)


def _spawn_dummy(tmp_path: Path, sandbox: _Sandbox) -> subprocess.Popen:
    """A real process the stub reports as a match — it must actually die."""
    dummy = subprocess.Popen(["sleep", "300"])
    Path(sandbox.env["PGREP_DUMMY_PID_FILE"]).write_text(f"{dummy.pid}\n", encoding="utf-8")
    return dummy


def _assert_killed(dummy: subprocess.Popen) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if dummy.poll() is not None:
            return
        time.sleep(0.05)
    dummy.kill()
    raise AssertionError(
        "the pattern-matching dummy process was not killed: the launcher's "
        "kill helper stopped selecting victims entirely"
    )


def _restart(sandbox: _Sandbox, reason: str) -> subprocess.CompletedProcess[str]:
    return _run(
        sandbox,
        "--foreground",
        "--restart",
        "--reason",
        reason,
        "--port",
        str(_free_port()),
    )


def test_restart_with_pattern_in_reason_does_not_kill_the_launcher(tmp_path):
    """D2: a --reason mentioning llama-server must not SIGTERM the launcher."""
    sandbox = _sandbox(tmp_path, via_symlink=False)
    dummy = _spawn_dummy(tmp_path, sandbox)
    try:
        proc = _restart(sandbox, "operator mitigation forced llama-server restart")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(_uvicorn_argv(sandbox)) == 1, (
            f"the launcher died before the start phase — it SIGTERMed itself: {proc.stdout + proc.stderr}"
        )
        _assert_killed(dummy)
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait(timeout=5)


def test_pid_guard_protects_a_renamed_launcher(tmp_path):
    """D2: the pid check alone must protect a launcher called ``launcher.sh``.

    The command-line exclusion matches ``start-proxy.sh``; invoking through a
    symlink removes that safety net, so this case fails unless the launcher's
    own pid is filtered out before any signal is sent.
    """
    sandbox = _sandbox(tmp_path, via_symlink=True)
    dummy = _spawn_dummy(tmp_path, sandbox)
    try:
        proc = _restart(sandbox, "rename probe: llama-server and uvicorn proxy.server")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(_uvicorn_argv(sandbox)) == 1, (
            f"the launcher SIGTERMed itself despite the pid guard: {proc.stdout + proc.stderr}"
        )
        _assert_killed(dummy)
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait(timeout=5)


def test_restart_never_reaches_the_live_port_kill_fallback(tmp_path):
    """LP-0MW3PLEH8008O4I7: a sandboxed restart must not kill by port.

    Phase 3 (``fuser -k 8080/8081``) selects victims by *port*, so no ``pgrep``
    stub can sandbox it: run for real it SIGKILLs the live TTS server and the
    live llama-server router. With ``START_PROXY_PORT_KILL_DISABLED=1`` the
    phase is skipped and ``fuser`` is never executed at all.
    """
    sandbox = _sandbox(tmp_path, via_symlink=False)
    proc = _restart(sandbox, "guard probe: llama-server and tts-server")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Port-kill fallback disabled by START_PROXY_PORT_KILL_DISABLED=1." in proc.stdout
    assert not sandbox.fuser_log.exists(), (
        "the port-kill fallback ran despite the guard: "
        f"{sandbox.fuser_log.read_text(encoding='utf-8') if sandbox.fuser_log.exists() else ''}"
    )


def test_port_kill_fallback_still_runs_when_the_guard_is_unset(tmp_path):
    """The guard must not change production behaviour.

    Without ``START_PROXY_PORT_KILL_DISABLED`` phase 3 still runs and targets the
    expected ports (``8080/tcp`` when llama-server is not preserved, always
    ``8081/tcp``); the stub records the invocation instead of signalling a host
    process.
    """
    sandbox = _sandbox(tmp_path, via_symlink=False, port_kill_disabled=False)
    proc = _restart(sandbox, "default behaviour probe: llama-server restart")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Port-kill fallback disabled" not in proc.stdout
    recorded = sandbox.fuser_log.read_text(encoding="utf-8").split("\n")
    assert "-k 8080/tcp" in recorded
    assert "-k 8081/tcp" in recorded
