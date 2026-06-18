import os
import subprocess
import textwrap
import socket
from pathlib import Path


def _find_free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    addr, port = s.getsockname()
    s.close()
    return port


def test_start_background_writes_pid_and_health(tmp_path):
    # Prepare proxy dir and start script
    tmp_proxy = tmp_path / 'proxy'
    tmp_proxy.mkdir()
    scripts = tmp_proxy / 'scripts'
    scripts.mkdir()
    port = _find_free_port()

    start_sh = scripts / 'start-proxy.sh'
    start_sh.write_text(textwrap.dedent(f"""#!/usr/bin/env bash
python3 - <<PY
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(os.environ.get('LLAMA_SERVER_PORT', {port}))
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        return

server = HTTPServer(('127.0.0.1', port), H)
server.serve_forever()
PY
"""))
    start_sh.chmod(0o755)

    # Copy proxyctl script into temp proxy dir
    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_proxy / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    # Use XDG_RUNTIME_DIR so pid/logs are written under tmp_path (preferred)
    env['XDG_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')
    env['LLAMA_SERVER_PORT'] = str(port)

    # Start background
    res = subprocess.run([str(script_dst), 'start'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=60)
    assert res.returncode == 0

    pid_file = Path(env['XDG_STATE_HOME']) / 'llama-proxy' / 'proxy.pid'
    assert pid_file.exists()
    pid = int(pid_file.read_text().strip())

    # Health probe should be reachable
    import time, http.client
    for _ in range(10):
        try:
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
            conn.request('GET', '/health')
            r = conn.getresponse()
            assert r.status == 200
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise AssertionError('Health endpoint did not respond')

    # Stop via proxyctl
    res2 = subprocess.run([str(script_dst), 'stop'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=10)
    assert res2.returncode == 0


def test_start_foreground_runs_and_exits(tmp_path):
    tmp_proxy = tmp_path / 'proxy'
    tmp_proxy.mkdir()
    scripts = tmp_proxy / 'scripts'
    scripts.mkdir()

    # Start script that prints and exits 0
    start_sh = scripts / 'start-proxy.sh'
    start_sh.write_text('#!/usr/bin/env bash\necho started\nexit 0\n')
    start_sh.chmod(0o755)

    base_dir = Path(__file__).resolve().parent.parent
    script_src = base_dir / 'proxyctl'
    script_dst = tmp_proxy / 'proxyctl'
    script_dst.write_bytes(script_src.read_bytes())
    script_dst.chmod(0o755)

    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['XDG_STATE_HOME'] = str(tmp_path / 'state')

    res = subprocess.run([str(script_dst), 'start', '--foreground'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=10)
    assert res.returncode == 0
    assert 'started' in res.stdout


def test_start_missing_start_script_returns_error(tmp_path):
    # No scripts/start-proxy.sh present
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

    res = subprocess.run([str(script_dst), 'start'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    assert res.returncode != 0
    assert 'Start script not found' in res.stderr
