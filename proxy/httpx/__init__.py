"""Local httpx shim for tests.

Exports an AsyncClient that supports `app=` for ASGI testing and basic
blocking fallbacks for simple requests. This file intentionally shadows
the installed `httpx` package during tests so the test suite runs
consistently across environments.
"""
from typing import Any, Optional, Callable, Iterable, Mapping, Union
import asyncio
import inspect
import sys

try:
    from starlette.testclient import TestClient as _StarletteTestClient
except Exception:
    _StarletteTestClient = None

import requests
from requests import Response as _RequestsResponse


class Response:
    def __init__(self, status_code: int = 200, content: bytes = b"", headers: Optional[dict] = None, json_data: Any = None):
        self.status_code = status_code
        self._content = content
        self._headers = headers or {}
        self._json_data = json_data

    @property
    def headers(self) -> dict:
        return self._headers

    @property
    def content(self) -> bytes:
        return self._content

    @property
    def text(self) -> str:
        return self._content.decode("utf-8", errors="replace")

    def json(self):
        if self._json_data is not None:
            return self._json_data
        import json as _json
        return _json.loads(self._content.decode("utf-8"))

    @property
    def status_code_prop(self) -> int:
        return self.status_code


class SimpleResponse:
    def __init__(self, resp: _RequestsResponse):
        self._resp = resp

    @property
    def status_code(self) -> int:
        return self._resp.status_code

    @property
    def headers(self) -> dict:
        return dict(self._resp.headers)

    @property
    def text(self) -> str:
        return self._resp.text

    @property
    def content(self) -> bytes:
        return self._resp.content

    def json(self):
        return self._resp.json()


class AsyncClient:
    def __init__(self, *args, **kwargs):
        app = kwargs.pop('app', None)
        base_url = kwargs.pop('base_url', "")
        timeout = kwargs.pop('timeout', None)

        if not app and len(args) >= 1:
            app = args[0]
        if not base_url and len(args) >= 2:
            base_url = args[1]
        if timeout is None and len(args) >= 3:
            timeout = args[2]

        self._app = app
        self.base_url = base_url or ""
        self._test_client = None
        self._timeout = timeout

    async def __aenter__(self):
        if self._app is not None:
            if _StarletteTestClient is not None:
                self._test_client = _StarletteTestClient(self._app)
            else:
                self._test_client = _ASGIClient(self._app)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._test_client is not None:
            self._test_client.close()
            self._test_client = None

    async def aclose(self):
        if self._test_client is not None:
            self._test_client.close()
            self._test_client = None

    async def get(self, url: str, **kwargs):
        if self._test_client is not None:
            return self._test_client.get(url, **kwargs)
        resp = requests.get(url, timeout=self._timeout or kwargs.get('timeout'))
        return SimpleResponse(resp)

    async def post(self, url: str, json: Any = None, data: Any = None, **kwargs):
        if self._test_client is not None:
            return self._test_client.post(url, json=json, data=data, **kwargs)
        resp = requests.post(url, json=json, data=data, timeout=self._timeout or kwargs.get('timeout'))
        return SimpleResponse(resp)

    def stream(self, method: str, url: str, headers: Any = None, content: Any = None, **kwargs):
        if self._test_client is not None:
            return self._test_client.stream(method, url, headers=headers, content=content, **kwargs)
        class CM:
            def __init__(self, client, method, url, headers, content):
                self.client = client
                self.method = method
                self.url = url
                self.headers = headers
                self.content = content

            async def __aenter__(self):
                resp = requests.request(self.method, self.url, headers=self.headers, data=self.content, stream=True, timeout=self.client._timeout)
                return SimpleResponse(resp)

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return CM(self, method, url, headers, content)


class _ASGIClient:
    """Minimal ASGI test client that doesn't require starlette."""
    def __init__(self, app: Callable):
        self.app = app
        self._scope = None
        self._response_started = False
        self._response_body = []

    def get(self, url: str, **kwargs):
        return self._request("GET", url, **kwargs)

    def post(self, url: str, json=None, data=None, **kwargs):
        return self._request("POST", url, json=json, data=data, **kwargs)

    def request(self, method: str, url: str, headers: Optional[Mapping] = None, json: Any = None, data: Any = None, **kwargs):
        return self._request(method, url, headers=headers, json=json, data=data, **kwargs)

    async def _asgi_handle(self, scope):
        body = b""
        if self._json_data is not None:
            import json as _json
            body = _json.dumps(self._json_data).encode()
        elif self._post_data is not None:
            body = str(self._post_data).encode()

        body_bytes = [body]
        response_started = [False]
        response_headers = [{}]
        response_status = [200]

        async def receive():
            return {"type": "http.request", "body": body_bytes[0] if body_bytes else b""}

        async def send(message):
            if message["type"] == "http.response.start":
                response_started[0] = True
                response_headers[0] = {k.decode(): v.decode() for k, v in message.get("headers", [])}
                response_status[0] = message.get("status", 200)
            elif message["type"] == "http.response.body":
                body_bytes.append(message.get("body", b""))

        await self.app(scope, receive, send)

        return Response(
            status_code=response_status[0],
            content=b"".join(body_bytes),
            headers=response_headers[0],
        )

    def _request(self, method: str, url: str, headers: Optional[Mapping] = None, json: Any = None, data: Any = None, **kwargs):
        if headers is None:
            headers = {}
        path = url.split("?")[0]
        query_string = url.split("?", 1)[1] if "?" in url else ""
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": query_string.encode(),
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            "server": ("testserver", 80),
            "asgi": {"version": "3.0"},
        }
        self._json_data = json
        self._post_data = data

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._asgi_handle(scope))
                return future.result()
        else:
            return asyncio.run(self._asgi_handle(scope))

    def close(self):
        pass

    def stream(self, method, url, headers=None, content=None, **kwargs):
        class CM:
            def __init__(self, client, method, url, headers, content):
                self.client = client
                self.method = method
                self.url = url
                self.headers = headers
                self.content = content

            async def __aenter__(self):
                return self.client._request(self.method, self.url, headers=self.headers, json=None, data=self.content)

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return CM(self, method, url, headers, content)
