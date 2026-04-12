import sys
import os

proxy_dir = os.path.join(os.getcwd())
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)

print(f"[conftest] sys.path[0]={repr(sys.path[0])}")
print(f"[conftest] cwd={os.getcwd()}")
print(f"[conftest] '' in sys.path: {'' in sys.path}")
print(f"[conftest] httpx in sys.modules: {'httpx' in sys.modules}")
if 'httpx' in sys.modules:
    print(f"[conftest] sys.modules['httpx'].__file__={getattr(sys.modules['httpx'],'__file__','N/A')}")

import importlib.util
repo_httpx_init = os.path.join(os.getcwd(), "httpx", "__init__.py")
if os.path.exists(repo_httpx_init):
    spec = importlib.util.spec_from_file_location("httpx", repo_httpx_init)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules['httpx'] = module
        spec.loader.exec_module(module)
        print(f"[conftest] Replaced httpx with local shim. AsyncClient={module.AsyncClient}")

