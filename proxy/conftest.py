import sys
import os

proxy_dir = os.path.join(os.getcwd())
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)

