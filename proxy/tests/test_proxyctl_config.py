import os
import subprocess
import textwrap
from pathlib import Path


def test_env_override_start_script(tmp_path):
    base_dir = Path(__file__).resolve().parent.parent
    script = base_dir / 'proxyctl'
    env = os.environ.copy()
    env['LLAMA_START_SCRIPT'] = '/tmp/from_env.sh'
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')
    env['HOME'] = str(tmp_path / 'home')

    res = subprocess.run([str(script), 'debug-resolve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res.returncode == 0
    assert 'start_script=/tmp/from_env.sh' in res.stdout


def test_config_start_script(tmp_path):
    # create temp proxy dir with config.yaml and copied proxyctl
    tmp_proxy = tmp_path / 'proxy'
    tmp_proxy.mkdir()
    config = tmp_proxy / 'config.yaml'
    config.write_text(textwrap.dedent('''
    server:
      llama_start_script: /tmp/from_config.sh
    '''))

    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_proxy / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env.pop('LLAMA_START_SCRIPT', None)
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')
    env['HOME'] = str(tmp_path / 'home')

    res = subprocess.run([str(script_dst), 'debug-resolve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res.returncode == 0
    assert 'start_script=/tmp/from_config.sh' in res.stdout


def test_default_fallback(tmp_path):
    # create temp proxy dir with scripts/start-proxy.sh and copied proxyctl
    tmp_parent = tmp_path / 'parent'
    tmp_parent.mkdir()
    tmp_proxy = tmp_parent / 'proxy'
    tmp_proxy.mkdir()
    scripts = tmp_proxy / 'scripts'
    scripts.mkdir()
    start_script = scripts / 'start-proxy.sh'
    start_script.write_text('#!/usr/bin/env bash\n')
    start_script.chmod(0o755)

    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_proxy / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env.pop('LLAMA_START_SCRIPT', None)
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')
    env['HOME'] = str(tmp_path / 'home')

    res = subprocess.run([str(script_dst), 'debug-resolve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res.returncode == 0
    expected_path = str(start_script.resolve())
    assert f'start_script={expected_path}' in res.stdout
