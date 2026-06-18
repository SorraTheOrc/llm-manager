import os
import subprocess
import time
from pathlib import Path


def test_logs_file_override_and_lines(tmp_path):
    lf = tmp_path / 'my.log'
    lf.write_text('\n'.join([f'line {i}' for i in range(1, 21)]))

    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_path / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')

    res = subprocess.run([str(script_dst), 'logs', '--file', str(lf), '--lines', '5'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=10)
    assert res.returncode == 0
    out_lines = res.stdout.strip().splitlines()
    assert out_lines[-1].endswith('line 20')
    assert len(out_lines) == 5


def test_logs_auto_detect(tmp_path):
    runtime = tmp_path / 'runtime'
    log_dir = runtime / 'llama-proxy' / 'logs'
    log_dir.mkdir(parents=True)
    lf = log_dir / 'proxy.log'
    lf.write_text('\n'.join([f'entry {i}' for i in range(1, 11)]))

    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_path / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = str(runtime)
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')

    res = subprocess.run([str(script_dst), 'logs', '--lines', '3'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=10)
    assert res.returncode == 0
    out_lines = res.stdout.strip().splitlines()
    assert out_lines[-1].endswith('entry 10')
    assert len(out_lines) == 3


def test_logs_missing_or_unreadable(tmp_path):
    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_path / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')

    # Missing file
    res = subprocess.run([str(script_dst), 'logs', '--file', str(tmp_path / 'nofile.log')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res.returncode != 0
    assert 'Logfile does not exist' in res.stdout

    # Unreadable file
    lf = tmp_path / 'unread.log'
    lf.write_text('secret')
    lf.chmod(0o000)
    res2 = subprocess.run([str(script_dst), 'logs', '--file', str(lf)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res2.returncode != 0
    assert 'Logfile is not readable' in res2.stderr
