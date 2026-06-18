import os
import subprocess
import time
import signal
from pathlib import Path


def test_stop_terminates_on_sigterm(tmp_path):
    # Start a simple process that exits on SIGTERM (sleep is fine)
    proc = subprocess.Popen(["sleep", "300"])
    try:
        runtime = tmp_path / 'runtime'
        pid_dir = runtime / 'llama-proxy'
        pid_dir.mkdir(parents=True)
        pid_file = pid_dir / 'proxy.pid'
        pid_file.write_text(str(proc.pid))

        # Copy proxyctl into temp dir
        base_dir = Path(__file__).resolve().parent.parent
        script_src = base_dir / 'proxyctl'
        script_dst = tmp_path / 'proxyctl'
        script_dst.write_bytes(script_src.read_bytes())
        script_dst.chmod(0o755)

        env = os.environ.copy()
        env['XDG_RUNTIME_DIR'] = str(runtime)
        env['XDG_STATE_HOME'] = str(tmp_path / 'state')

        res = subprocess.run([str(script_dst), 'stop'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        assert res.returncode == 0
        assert not pid_file.exists()
        # process should be gone
        proc.poll()
        assert proc.returncode is not None
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_stop_escalates_to_kill_and_returns_nonzero(tmp_path):
    # Start a process that ignores SIGTERM
    ignore_sh = tmp_path / 'ignore.sh'
    ignore_sh.write_text('#!/usr/bin/env bash\ntrap "" TERM\nwhile true; do sleep 1; done\n')
    ignore_sh.chmod(0o755)

    proc = subprocess.Popen([str(ignore_sh)])
    try:
        runtime = tmp_path / 'runtime'
        pid_dir = runtime / 'llama-proxy'
        pid_dir.mkdir(parents=True)
        pid_file = pid_dir / 'proxy.pid'
        pid_file.write_text(str(proc.pid))

        base_dir = Path(__file__).resolve().parent.parent
        script_src = base_dir / 'proxyctl'
        script_dst = tmp_path / 'proxyctl'
        script_dst.write_bytes(script_src.read_bytes())
        script_dst.chmod(0o755)

        env = os.environ.copy()
        env['XDG_RUNTIME_DIR'] = str(runtime)
        env['XDG_STATE_HOME'] = str(tmp_path / 'state')

        res = subprocess.run([str(script_dst), 'stop'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        # Expect non-zero indicating forced kill
        assert res.returncode != 0
        # pidfile removed
        assert not pid_file.exists()
        # process should be dead
        proc.wait(timeout=2)
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_stop_no_pidfile_returns_zero(tmp_path):
    tmp_proxy = tmp_path / 'proxy'
    tmp_proxy.mkdir()
    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_proxy / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')

    res = subprocess.run([str(script_dst), 'stop'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert 'proxy not running' in res.stdout
