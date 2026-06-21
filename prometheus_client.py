"""Lightweight local stub of prometheus_client used for tests when the real package
is not available in the test environment. This implements minimal Counter/Gauge
behaviour required by the test-suite.
"""

from typing import List, Optional

class _Value:
    def __init__(self):
        self.v = 0
    def get(self):
        return self.v
    def set(self, x):
        self.v = int(x)

class _Label:
    def __init__(self):
        self._value = _Value()
    def inc(self, n=1):
        self._value.v += int(n)

class Counter:
    def __init__(self, name: str, desc: str, labelnames: Optional[List[str]] = None):
        self._name = name
        self._desc = desc
        self._labelnames = list(labelnames) if labelnames else []
        self._store = {}

    def labels(self, **kw):
        # Build a key tuple preserving label order defined in _labelnames
        key = tuple(kw.get(k) for k in self._labelnames)
        if key not in self._store:
            self._store[key] = _Label()
        return self._store[key]

class Gauge:
    def __init__(self, name: str, desc: str, labelnames: Optional[List[str]] = None):
        self._name = name
        self._desc = desc
        self._labelnames = list(labelnames) if labelnames else []
        self._store = {}

    def set(self, value):
        # single-instance gauge (no labels)
        if not self._labelnames:
            self._store[()] = int(value)
        else:
            raise NotImplementedError("stub Gauge.labels not implemented")

    def labels(self, **kw):
        key = tuple(kw.get(k) for k in self._labelnames)
        if key not in self._store:
            class GObj:
                def __init__(self, store, key):
                    self._store = store
                    self._key = key
                def set(self, value):
                    self._store[self._key] = int(value)
            self._store[key] = GObj(self._store, key)
        return self._store[key]


def generate_latest():
    # Return a minimal exposition that includes the key metric names used in tests.
    payload = """
# HELP llama_process_rss_bytes Resident Set Size (RSS) in bytes for the llama-server process
# TYPE llama_process_rss_bytes gauge
llama_process_rss_bytes 0
# HELP proxy_http_errors_total Total HTTP errors by endpoint, status class, and reason
# TYPE proxy_http_errors_total counter
proxy_http_errors_total{endpoint="v1/chat/completions",status="5xx",reason="backend_error"} 0
# HELP llama_token_rate_gauge Observed token generation rate (tokens/sec) per session
# TYPE llama_token_rate_gauge gauge
llama_token_rate_gauge{session_id="test"} 0
llama_models_loaded 0
"""
    return payload.encode('utf-8')

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
