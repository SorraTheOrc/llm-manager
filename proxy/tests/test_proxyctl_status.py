import os
import subprocess
import time
from pathlib import Path


def test_status_when_running(tmp_path):
    # Start a simple process that will remain running (sleep)
    proc = subprocess.Popen(["sleep", "300"])  # ensure process will not exit during test
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

        res = subprocess.run([str(script_dst), 'status'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        assert res.returncode == 0
        assert f"proxy running (PID {proc.pid})" in res.stdout
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_status_when_not_running(tmp_path):
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

    res = subprocess.run([str(script_dst), 'status'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    assert res.returncode == 3
    assert 'proxy not running' in res.stdout
