"""
Observability Module

Centralizes backend signal tracking (connect/read/timeout/other failures),
SSE client management for real-time status and log tail broadcasts, and
exception classification for observability.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
from typing import Any
import json
import time
from pathlib import Path
import httpx


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backend signal counters (connect/read/timeout/other/concurrency)
# ===================================================================

backend_signal_counts: dict = {
    "connect_failures": 0,
    "read_failures": 0,
    "timeout_failures": 0,
    "other_failures": 0,
    "concurrency_rejects": 0,
}


def _record_backend_signal(signal_name: str) -> None:
    """Increment a backend signal counter for observability."""
    srv = _srv()
    try:
        if signal_name in srv.backend_signal_counts:
            srv.backend_signal_counts[signal_name] = (
                int(srv.backend_signal_counts.get(signal_name, 0)) + 1
            )
    except Exception:
        pass


def _classify_backend_exception(exc: Exception) -> str:
    """Map backend transport exceptions to signal buckets."""
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "connect_failures"
    if isinstance(exc, (httpx.ReadError,)):
        return "read_failures"
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return "timeout_failures"
    return "other_failures"


# ===================================================================
# SSE clients for real-time broadcasts
# ===================================================================

sse_clients: set[asyncio.Queue] = set()
log_tail_clients: set[asyncio.Queue] = set()


# ===================================================================
# Counters, tokens, status broadcast and persistence
# ===================================================================

async def broadcast_status(event_type: str, data: dict):
    """Broadcast a status event to all connected SSE clients."""
    srv = _srv()
    event_data = json.dumps({"type": event_type, **data})
    message = f"data: {event_data}\n\n"
    
    # Send to all connected clients
    dead_clients = set()
    for queue in sse_clients:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            dead_clients.add(queue)
    
    # Clean up dead clients
    for client in dead_clients:
        sse_clients.discard(client)



def broadcast_status_sync(event_type: str, data: dict):
    """Synchronous wrapper to broadcast status (for use in sync code)."""
    srv = _srv()
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_status(event_type, data))
    except RuntimeError:
        # No running loop, skip broadcast
        pass



def _counts_file_path() -> Path:
    """Return path to the persisted counts file inside log_dir (or local logs)."""
    srv = _srv()
    if srv.log_dir:
        return srv.log_dir / srv.counts_filename
    return Path(__file__).parent / "logs" / srv.counts_filename



def load_counts():
    srv = _srv()
    try:
        path = _counts_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                srv.request_counts = json.load(f)
        else:
            srv.request_counts = {}
    except Exception:
        srv.request_counts = {}



def save_counts_sync():
    srv = _srv()
    try:
        path = _counts_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(request_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        srv.logger.error(f"Failed to persist request counts: {e}")



def _token_file_path() -> Path:
    srv = _srv()
    if srv.log_dir:
        return srv.log_dir / srv.token_counts_filename
    return Path(__file__).parent / "logs" / srv.token_counts_filename



def load_token_counts():
    srv = _srv()
    try:
        path = _token_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                srv.token_counts = json.load(f)
        else:
            srv.token_counts = {}
    except Exception:
        srv.token_counts = {}



def save_token_counts_sync():
    srv = _srv()
    try:
        path = _token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(token_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        srv.logger.error(f"Failed to persist token counts: {e}")



async def save_token_counts():
    srv = _srv()
    await asyncio.to_thread(save_token_counts_sync)



async def save_counts():
    srv = _srv()
    await asyncio.to_thread(save_counts_sync)



async def _counts_persist_loop():
    srv = _srv()
    try:
        while True:
            await asyncio.sleep(2.0)
            if srv.counts_dirty:
                try:
                    await save_counts()
                    srv.counts_dirty = False
                except Exception:
                    pass
    finally:
        srv.counts_persist_task = None



async def _tokens_persist_loop():
    srv = _srv()
    try:
        while True:
            await asyncio.sleep(2.0)
            if srv.tokens_dirty:
                try:
                    await save_token_counts()
                    srv.tokens_dirty = False
                except Exception:
                    pass
    finally:
        srv.tokens_persist_task = None



async def query_llama_status() -> dict:
    """
    Query llama-server HTTP endpoints for model metadata.

    Attempts HTTP GET to /model then /status and returns parsed JSON if successful.
    If those endpoints are not present or do not include n_ctx / KV cache values,
    returns null for those fields.

    Returns dict with:
      - n_ctx: max context size (int) or None
      - kv_cache_tokens: KV cache token count (int) or None
      - llama_server_running: bool
      - router_mode: bool
    """
    # allow updating/reading endpoint cache/failures
    srv = _srv()

    result = {
        "n_ctx": None,
        "kv_cache_tokens": None,
        "llama_server_running": srv.llama_process is not None and srv.llama_process.poll() is None,
        "router_mode": False
    }

    if not result["llama_server_running"]:
        return result

    server_config = srv.config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)

    # One-time discovery: probe metadata endpoints once per llama-server process
    client = srv._http_client if srv._http_client else httpx.AsyncClient(timeout=5.0)
    try:
        async def _do_discovery_if_needed():
            """Discover a working metadata endpoint once per process.

            Returns a (n_ctx, kv_cache_tokens) tuple discovered during probing
            so callers can use the data without re-requesting mocked responses
            (important for tests that provide a sequence of responses).
            """
            # If discovery was already done for this pid, skip.
            try:
                current_pid = getattr(llama_process, 'pid', None)
            except Exception:
                current_pid = None
            if srv._llama_status_discovered and srv._llama_status_discovered_pid == current_pid:
                return None, None

            endpoints = ["/model", "/status", "/models", "/v1/models"]
            found_n = None
            found_kv = None
            for endpoint in endpoints:
                try:
                    url = f"http://localhost:{llama_port}{endpoint}"
                    resp = await client.get(url, timeout=5.0)
                    if getattr(resp, 'status_code', None) == 200:
                        # remember endpoint
                        if not srv._llama_status_endpoint_cache:
                            srv._llama_status_endpoint_cache = endpoint
                        srv._llama_status_endpoint_failures.pop(endpoint, None)

                        # attempt to parse JSON/text for n_ctx / kv fields
                        data = None
                        if hasattr(resp, 'json'):
                            try:
                                maybe = resp.json()
                                data = await maybe if asyncio.iscoroutine(maybe) else maybe
                            except Exception:
                                data = None
                        if data is None and hasattr(resp, 'text'):
                            try:
                                txt = resp.text if not asyncio.iscoroutine(resp.text) else await resp.text
                                data = json.loads(txt)
                            except Exception:
                                data = None

                        if isinstance(data, dict):
                            if found_n is None:
                                found_n = data.get("n_ctx") or data.get("n_ctx_total")
                            if found_kv is None:
                                found_kv = data.get("kv_cache_tokens") or data.get("kv_cache_token_count")

                        # If we've discovered both values, stop probing
                        if found_n is not None and found_kv is not None:
                            break
                except Exception:
                    # ignore and try next
                    continue

            # Mark discovery done for this pid even if none found
            srv._llama_status_discovered = True
            try:
                srv._llama_status_discovered_pid = getattr(llama_process, 'pid', None)
            except Exception:
                srv._llama_status_discovered_pid = None

            return found_n, found_kv

        # Run discovery and capture any discovered metadata so we can
        # return parsed values immediately (important for tests that
        # provide a sequence of mocked responses consumed during discovery).
        found_n, found_kv = await _do_discovery_if_needed()
        if found_n is not None:
            result["n_ctx"] = found_n
        if found_kv is not None:
            result["kv_cache_tokens"] = found_kv

        # If we have a cached endpoint, try it first
        if srv._llama_status_endpoint_cache:
            try:
                url = f"http://localhost:{llama_port}{srv._llama_status_endpoint_cache}"
                response = await client.get(url, timeout=5.0)
                if getattr(response, 'status_code', None) == 200:
                    data = None
                    if hasattr(response, 'json'):
                        try:
                            maybe = response.json()
                            data = await maybe if asyncio.iscoroutine(maybe) else maybe
                        except Exception:
                            data = None
                    if data is None and hasattr(response, 'text'):
                        try:
                            txt = response.text if not asyncio.iscoroutine(response.text) else await response.text
                            data = json.loads(txt)
                        except Exception:
                            data = None

                    if isinstance(data, dict):
                        if result["n_ctx"] is None:
                            result["n_ctx"] = data.get("n_ctx") or data.get("n_ctx_total")
                        if result["kv_cache_tokens"] is None:
                            result["kv_cache_tokens"] = (
                                data.get("kv_cache_tokens") or data.get("kv_cache_token_count")
                            )
                else:
                    # cache no longer valid; clear and allow future rediscovery
                    srv._llama_status_endpoint_failures[_llama_status_endpoint_cache] = time.time()
                    srv._llama_status_endpoint_cache = None
                    srv._llama_status_discovered = False
                    srv._llama_status_discovered_pid = None
            except Exception:
                # ignore and fallthrough to props check
                pass

        # As a fallback, check /props to detect router mode
        try:
            props_url = f"http://localhost:{llama_port}/props"
            response = await client.get(props_url, timeout=5.0)
            if getattr(response, "status_code", None) == 200:
                props = None
                if hasattr(response, "json"):
                    try:
                        maybe = response.json()
                        props = await maybe if asyncio.iscoroutine(maybe) else maybe
                    except Exception:
                        props = None
                if props is None and hasattr(response, "text"):
                    try:
                        txt = response.text if not asyncio.iscoroutine(response.text) else await response.text
                        props = json.loads(txt)
                    except Exception:
                        props = None
                if isinstance(props, dict):
                    result["router_mode"] = True
        except Exception:
            pass
    finally:
        if not srv._http_client:
            try:
                await client.aclose()
            except Exception:
                pass

    return result



async def _periodic_broadcast_loop():
    """Periodically broadcast current counts/tokens to connected log-tail clients.

    This ensures UI updates even if no direct increment message races occur.
    Also queries llama-server for status and broadcasts stats.
    """
    srv = _srv()
    try:
        while True:
            try:
                await asyncio.sleep(1.0)
                snap_c = {}
                snap_t = {}
                async with srv.counts_lock:
                    snap_c = dict(srv.request_counts)
                async with srv.token_lock:
                    snap_t = dict(srv.token_counts)

                llama_status = await query_llama_status()
                
                total_sent = srv.token_counts.get("total_sent", 0)
                total_recv = srv.token_counts.get("total_recv", 0)

                if log_tail_clients:
                    for q in list(log_tail_clients):
                        try:
                            q.put_nowait({
                                "counts": snap_c, 
                                "tokens": snap_t,
                                "llama_status": llama_status,
                                "total_sent": total_sent,
                                "total_recv": total_recv
                            })
                        except asyncio.QueueFull:
                            continue
                
                if sse_clients:
                    status_data = {
                        "type": "status",
                        "current_model": current_model,
                        "llama_server_running": llama_status["llama_server_running"],
                        "n_ctx": llama_status["n_ctx"],
                        "kv_cache_tokens": llama_status["kv_cache_tokens"],
                        "total_sent": total_sent,
                        "total_recv": total_recv
                    }
                    event_data = json.dumps(status_data)
                    message = f"data: {event_data}\n\n"
                    dead_clients = set()
                    for q in sse_clients:
                        try:
                            q.put_nowait(message)
                        except asyncio.QueueFull:
                            dead_clients.add(q)
                    for client in dead_clients:
                        sse_clients.discard(client)
                        
            except asyncio.CancelledError:
                break
            except Exception:
                # ignore transient errors
                pass
    finally:
        srv.periodic_broadcast_task = None



async def _increment_count(key: str):
    """Increment the in-memory counter for a request key and persist."""
    srv = _srv()
    try:
        async with srv.counts_lock:
            prev = srv.request_counts.get(key, 0)
            srv.request_counts[key] = prev + 1
            # mark dirty but don't persist immediately; background task will persist
            srv.counts_dirty = True
            srv.logger.debug(f"_increment_count: key={key} prev={prev} new={srv.request_counts[key]}")
    except Exception as e:
        srv.logger.error(f"Error incrementing request count: {e}")
    # Broadcast updated counts to connected log tail clients
    try:
        snapshot = None
        async with srv.counts_lock:
            snapshot = dict(srv.request_counts)

        # Send to all connected log-tail queues
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                # skip slow listeners
                continue
    except Exception:
        pass



async def _increment_count_multi(keys: list[str]):
    """Increment multiple request count keys in one go and broadcast snapshot."""
    srv = _srv()
    try:
        async with srv.counts_lock:
            # log previous values for debugging
            prevs = {k: srv.request_counts.get(k, 0) for k in keys}
            for key in keys:
                srv.request_counts[key] = prevs.get(key, 0) + 1
            srv.counts_dirty = True
            srv.logger.debug(f"_increment_count_multi: keys={keys} prevs={prevs} new_vals={{k: srv.request_counts[k] for k in keys}}")
    except Exception as e:
        srv.logger.error(f"Error incrementing request counts: {e}")

    # Broadcast snapshot
    try:
        snapshot = None
        async with srv.counts_lock:
            snapshot = dict(srv.request_counts)

        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass



async def _increment_tokens(key_prefix: str, key: str, n: int):
    """Increment token counts and persist; key_prefix is 'sent' or 'recv'."""
    srv = _srv()
    try:
        async with srv.token_lock:
            pk = key_prefix + ':' + key
            prev = srv.token_counts.get(pk, 0)
            srv.token_counts[pk] = prev + n
            total_key = 'total_sent' if key_prefix == 'sent' else 'total_recv'
            prev_total = srv.token_counts.get(total_key, 0)
            srv.token_counts[total_key] = prev_total + n
            srv.tokens_dirty = True
            srv.logger.debug(f"_increment_tokens: prefix={key_prefix} key={key} n={n} prev={prev} new={srv.token_counts[pk]} total_prev={prev_total} total_new={srv.token_counts[total_key]}")
    except Exception as e:
        srv.logger.error(f"Error incrementing token counts: {e}")
    # Broadcast token snapshot
    try:
        snap = None
        async with srv.token_lock:
            snap = dict(srv.token_counts)
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"tokens": snap})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass



