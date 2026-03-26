#!/usr/bin/env python3
"""
LLama Proxy Server

A proxy server that routes OpenAI API requests to either a local llama-server
or remote API services based on configuration.
"""

import asyncio
import json
import re
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from fnmatch import fnmatch
from typing import Any, Optional

import httpx
import shutil
import io
import traceback
from datetime import timedelta
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

# Global state
llama_process: Optional[subprocess.Popen] = None
llama_log_file: Optional[Any] = None
current_model: Optional[str] = None
last_start_failure: Optional[str] = None
model_switch_lock = asyncio.Lock()
# Request counting
request_counts: dict = {}
counts_lock = asyncio.Lock()
counts_filename = "request_counts.json"
# Dirty flags and persist tasks
counts_dirty = False
counts_persist_task: Optional[asyncio.Task] = None
periodic_broadcast_task: Optional[asyncio.Task] = None

# Token counting
token_counts: dict = {}
token_lock = asyncio.Lock()
token_counts_filename = "token_counts.json"
# Dirty flags and persist tasks for tokens
tokens_dirty = False
tokens_persist_task: Optional[asyncio.Task] = None


# Optional tokenizer (tiktoken)
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

def _get_tiktoken_encoding_for_model(model_name: str | None):
    if not tiktoken:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_text_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens in text using tiktoken if available, otherwise a heuristic."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding_for_model(model_name)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # heuristic: 1 token ~ 4 bytes UTF-8
    return max(1, len(text.encode('utf-8')) // 4)
config: dict = {}
log_dir: Optional[Path] = None
logger: logging.Logger = logging.getLogger("llama-proxy")

# SSE clients for real-time status updates
sse_clients: set[asyncio.Queue] = set()
# SSE clients for log tail updates (counts + other notifications)
log_tail_clients: set[asyncio.Queue] = set()


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "LLAMA_PROXY_CONFIG",
            str(Path(__file__).parent / "config.yaml")
        )
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging with time-based rotation."""
    global log_dir
    
    log_config = config.get("logging", {})
    log_dir = Path(log_config.get("directory", "/var/log/llama-proxy"))
    rotation_hours = log_config.get("rotation_hours", 6)
    retention_days = log_config.get("retention_days", 90)
    log_level = log_config.get("level", "INFO")
    
    # Try to create log directory, fall back to local logs directory if permission denied
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using local log directory: {log_dir}")
    
    # Calculate backup count based on retention days and rotation interval
    # (retention_days * 24 hours / rotation_hours)
    backup_count = (retention_days * 24) // rotation_hours
    
    # Create logger
    logger = logging.getLogger("llama-proxy")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with rotation
    log_file = log_dir / "proxy.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="H",  # Hourly rotation
        interval=rotation_hours,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)
    
    # Console handler for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(console_handler)
    
    return logger


async def broadcast_status(event_type: str, data: dict):
    """Broadcast a status event to all connected SSE clients."""
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
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_status(event_type, data))
    except RuntimeError:
        # No running loop, skip broadcast
        pass


def _counts_file_path() -> Path:
    """Return path to the persisted counts file inside log_dir (or local logs)."""
    if log_dir:
        return log_dir / counts_filename
    return Path(__file__).parent / "logs" / counts_filename


def load_counts():
    global request_counts
    try:
        path = _counts_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                request_counts = json.load(f)
        else:
            request_counts = {}
    except Exception:
        request_counts = {}


def save_counts_sync():
    try:
        path = _counts_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(request_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        logger.error(f"Failed to persist request counts: {e}")


def _token_file_path() -> Path:
    if log_dir:
        return log_dir / token_counts_filename
    return Path(__file__).parent / "logs" / token_counts_filename


def load_token_counts():
    global token_counts
    try:
        path = _token_file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                token_counts = json.load(f)
        else:
            token_counts = {}
    except Exception:
        token_counts = {}


def save_token_counts_sync():
    try:
        path = _token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(token_counts, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        logger.error(f"Failed to persist token counts: {e}")


async def save_token_counts():
    await asyncio.to_thread(save_token_counts_sync)


async def save_counts():
    await asyncio.to_thread(save_counts_sync)


async def _counts_persist_loop():
    global counts_dirty, counts_persist_task
    try:
        while True:
            await asyncio.sleep(2.0)
            if counts_dirty:
                try:
                    await save_counts()
                    counts_dirty = False
                except Exception:
                    pass
    finally:
        counts_persist_task = None


async def _tokens_persist_loop():
    global tokens_dirty, tokens_persist_task
    try:
        while True:
            await asyncio.sleep(2.0)
            if tokens_dirty:
                try:
                    await save_token_counts()
                    tokens_dirty = False
                except Exception:
                    pass
    finally:
        tokens_persist_task = None


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
    """
    result = {
        "n_ctx": None,
        "kv_cache_tokens": None,
        "llama_server_running": llama_process is not None and llama_process.poll() is None
    }
    
    if not result["llama_server_running"]:
        return result
    
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    
    client = httpx.AsyncClient(timeout=5.0)
    try:
        for endpoint in ["/model", "/status"]:
            try:
                url = f"http://localhost:{llama_port}{endpoint}"
                response = await client.get(url)
                if getattr(response, "status_code", None) == 200:
                    try:
                        # httpx Response.json may be sync or async in tests; handle both
                        data = None
                        if hasattr(response, 'json'):
                            try:
                                maybe = response.json()
                                if asyncio.iscoroutine(maybe):
                                    data = await maybe
                                else:
                                    data = maybe
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
                                    data.get("kv_cache_tokens") or 
                                    data.get("kv_cache_token_count")
                                )
                            if result["n_ctx"] is not None and result["kv_cache_tokens"] is not None:
                                break
                    except Exception:
                        pass
            except Exception:
                pass
    finally:
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
    global periodic_broadcast_task
    try:
        while True:
            try:
                await asyncio.sleep(1.0)
                snap_c = {}
                snap_t = {}
                async with counts_lock:
                    snap_c = dict(request_counts)
                async with token_lock:
                    snap_t = dict(token_counts)

                llama_status = await query_llama_status()
                
                total_sent = token_counts.get("total_sent", 0)
                total_recv = token_counts.get("total_recv", 0)

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
        periodic_broadcast_task = None


def detect_log_severity(line: str) -> Optional[str]:
    """Detect severity from a log line. Returns 'error'|'warning'|'info'|'debug' or None."""
    if not line:
        return None
    try:
        trimmed = line.strip()
        # Prefer JSON structured logs
        if trimmed.startswith('{'):
            try:
                j = json.loads(trimmed)
                lvl_raw = None
                for key in ('level', 'severity', 'levelname', 'level_name', 'levelName', 'loglevel'):
                    if key in j:
                        lvl_raw = j.get(key)
                        break
                if lvl_raw is not None:
                    lvl = str(lvl_raw).lower()
                    if 'err' in lvl:
                        return 'error'
                    if 'warn' in lvl:
                        return 'warning'
                    if 'info' in lvl:
                        return 'info'
                    if any(k in lvl for k in ('debug', 'trace', 'verb')):
                        return 'debug'
            except Exception:
                pass
        up = trimmed.upper()
        parts = re.split('[^A-Z]+', up)
        for p in parts:
            if not p:
                continue
            if p in ('ERROR', 'ERR'):
                return 'error'
            if p in ('WARN', 'WARNING'):
                return 'warning'
            if p == 'INFO':
                return 'info'
            if p in ('DEBUG', 'TRACE', 'VERB'):
                return 'debug'
        # fallback substring checks
        if ' - INFO - ' in up or '[INFO]' in up or re.search(r'LEVEL[:=]\s*INFO', up):
            return 'info'
        if ' - ERROR - ' in up or '[ERROR]' in up or re.search(r'LEVEL[:=]\s*ERROR', up):
            return 'error'
        if ' - WARN - ' in up or '[WARN]' in up or re.search(r'LEVEL[:=]\s*WARN', up):
            return 'warning'
        if ' - DEBUG - ' in up or '[DEBUG]' in up or re.search(r'LEVEL[:=]\s*DEBUG', up):
            return 'debug'
    except Exception:
        pass
    return None


async def _increment_count(key: str):
    """Increment the in-memory counter for a request key and persist."""
    try:
        async with counts_lock:
            prev = request_counts.get(key, 0)
            request_counts[key] = prev + 1
            # mark dirty but don't persist immediately; background task will persist
            global counts_dirty
            counts_dirty = True
            logger.debug(f"_increment_count: key={key} prev={prev} new={request_counts[key]}")
    except Exception as e:
        logger.error(f"Error incrementing request count: {e}")
    # Broadcast updated counts to connected log tail clients
    try:
        snapshot = None
        async with counts_lock:
            snapshot = dict(request_counts)

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
    try:
        async with counts_lock:
            # log previous values for debugging
            prevs = {k: request_counts.get(k, 0) for k in keys}
            for key in keys:
                request_counts[key] = prevs.get(key, 0) + 1
            global counts_dirty
            counts_dirty = True
            logger.debug(f"_increment_count_multi: keys={keys} prevs={prevs} new_vals={{k: request_counts[k] for k in keys}}")
    except Exception as e:
        logger.error(f"Error incrementing request counts: {e}")

    # Broadcast snapshot
    try:
        snapshot = None
        async with counts_lock:
            snapshot = dict(request_counts)

        for q in list(log_tail_clients):
            try:
                q.put_nowait({"counts": snapshot})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass


async def _increment_tokens(key_prefix: str, key: str, n: int):
    """Increment token counts and persist; key_prefix is 'sent' or 'recv'."""
    try:
        async with token_lock:
            pk = key_prefix + ':' + key
            prev = token_counts.get(pk, 0)
            token_counts[pk] = prev + n
            total_key = 'total_sent' if key_prefix == 'sent' else 'total_recv'
            prev_total = token_counts.get(total_key, 0)
            token_counts[total_key] = prev_total + n
            global tokens_dirty
            tokens_dirty = True
            logger.debug(f"_increment_tokens: prefix={key_prefix} key={key} n={n} prev={prev} new={token_counts[pk]} total_prev={prev_total} total_new={token_counts[total_key]}")
    except Exception as e:
        logger.error(f"Error incrementing token counts: {e}")
    # Broadcast token snapshot
    try:
        snap = None
        async with token_lock:
            snap = dict(token_counts)
        for q in list(log_tail_clients):
            try:
                q.put_nowait({"tokens": snap})
            except asyncio.QueueFull:
                continue
    except Exception:
        pass


def get_model_config(model_name: Optional[str]) -> Optional[dict]:
    """
    Get model configuration by name or alias.
    
    Supports wildcard patterns in aliases using fnmatch syntax:
    - '*' matches any sequence of characters
    - '?' matches any single character
    - '[seq]' matches any character in seq
    - '[!seq]' matches any character not in seq
    
    Examples:
    - 'gpt*' matches 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'
    - 'claude-3-*' matches 'claude-3-opus', 'claude-3-sonnet'
    """
    if model_name is None:
        return None
    
    models = config.get("models", {})
    
    # Direct match
    if model_name in models:
        return models[model_name]
    
    model_name_lower = model_name.lower()
    
    # Check exact aliases first (higher priority)
    for name, model_cfg in models.items():
        aliases = model_cfg.get("aliases", [])
        for alias in aliases:
            alias_lower = alias.lower()
            # Skip wildcard patterns in first pass
            if '*' in alias or '?' in alias or '[' in alias:
                continue
            if model_name_lower == alias_lower:
                return model_cfg
    
    # Check wildcard patterns
    for name, model_cfg in models.items():
        aliases = model_cfg.get("aliases", [])
        for alias in aliases:
            alias_lower = alias.lower()
            # Only process wildcard patterns
            if '*' in alias or '?' in alias or '[' in alias:
                if fnmatch(model_name_lower, alias_lower):
                    return model_cfg
    
    return None


def get_local_model_name(model_name: Optional[str]) -> Optional[str]:
    """Get the llama model name for a given model."""
    model_cfg = get_model_config(model_name)
    if model_cfg and model_cfg.get("type") == "local":
        return model_cfg.get("llama_model")
    return None


async def wait_for_llama_server(timeout: int = 300) -> bool:
    """Wait for llama-server to be ready."""
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    health_url = f"http://localhost:{llama_port}/health"
    
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            # Check if llama process died
            if llama_process is not None and llama_process.poll() is not None:
                exit_code = llama_process.returncode
                logger.error(f"llama-server process exited with code {exit_code}")
                return False
            
            try:
                response = await client.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("llama-server is ready")
                    return True
            except asyncio.CancelledError:
                logger.info("Wait for llama-server cancelled")
                raise
            except Exception:
                pass
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                logger.info("Wait for llama-server cancelled")
                raise
    
    logger.error(f"llama-server failed to start within {timeout} seconds")
    return False


def start_llama_server(model: str) -> Optional[subprocess.Popen]:
    """Start the llama-server with the specified model inside distrobox."""
    global llama_process, llama_log_file, current_model, last_start_failure
    
    server_config = config.get("server", {})
    # Default to the repository root `start-llama.sh` if not specified in config
    script_path = server_config.get(
        "llama_start_script",
        str(Path(__file__).parent.parent / "start-llama.sh")
    )
    distrobox_name = server_config.get("distrobox_name", "llama")
    llama_port = server_config.get("llama_server_port", 8080)
    
    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(llama_port)
    
    logger.info(f"Starting llama-server with model: {model} in distrobox '{distrobox_name}'")

    # Rotate llama-server logs (keep last 15)
    if log_dir:
        llama_log_path = log_dir / "llama-server.log"
        rotate_llama_logs(llama_log_path, keep=15)
        llama_log_file = open(llama_log_path, "w")
    else:
        llama_log_file = subprocess.DEVNULL

    # Try running via distrobox first; if distrobox is not available or fails,
    # fall back to running the start script directly. Capture and log errors
    # so failures after reboot are diagnosable. Implement a short retry loop
    # with backoff to tolerate boot-order races.
    distrobox_cmd = ["distrobox", "enter", distrobox_name, "--", script_path, model]
    direct_cmd = [script_path, model]

    # Helper to start a subprocess and capture immediate stderr/stdout if it
    # exits quickly. Returns a tuple (Popen|None, captured_output_str|None).
    def _spawn_and_capture(cmd):
        try:
            # Use pipes so we can capture early output if the child exits
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except FileNotFoundError as e:
            logger.warning(f"Command not found when starting llama-server: {cmd[0]}: {e}")
            return None, f"Command not found: {cmd[0]}: {e}"
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Failed to spawn command {cmd}: {e}\n{tb}")
            return None, f"Spawn failed: {e}\n{tb}"

        # Give the child a short window to produce output and check if it exits
        try:
            outs, _ = proc.communicate(timeout=3)
            # If communicate returned, the process exited quickly; include output
            return None, outs
        except subprocess.TimeoutExpired:
            # Process is still running — good.
            # Reattach stdout to file for long-running logging
            try:
                if llama_log_file and proc.stdout:
                    # Write the already-read portion (if any) and keep piping
                    # from proc.stdout asynchronously is complex; as a compromise
                    # we'll spawn a background thread to stream output to file.
                    import threading

                    def _stream_output(src, dst):
                        try:
                            for line in src:
                                dst.write(line)
                                dst.flush()
                        except Exception:
                            pass

                    t = threading.Thread(target=_stream_output, args=(proc.stdout, llama_log_file), daemon=True)
                    t.start()
            except Exception:
                pass
            return proc, None

    # Retry loop configuration
    retries = 4
    backoff = 2  # seconds base
    tried_cmds = []

    # The server MUST run inside distrobox/container. Do not fall back to
    # running the start script directly on the host. This avoids running
    # models outside a controlled container environment.
    if not shutil.which("distrobox"):
        msg = "distrobox not found in PATH; llama-server must be started inside distrobox"
        logger.error(msg)
        last_start_failure = msg
        broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
        return None

    # Try distrobox only
    for attempt in range(retries):
        proc, out = _spawn_and_capture(distrobox_cmd)
        tried_cmds.append((distrobox_cmd, out))
        if proc is not None:
            current_model = model
            return proc
        # If out is present, the command exited quickly with output; log and retry
        logger.warning(f"distrobox attempt {attempt+1} failed quickly: {out}")
        time.sleep(backoff * (attempt + 1))

    # All distrobox attempts failed — assemble a helpful diagnostic message
    msg_lines = ["Failed to start llama-server using distrobox. Attempts:"]
    for cmd, out in tried_cmds:
        cmd_str = " ".join(cmd)
        snippet = (out or "(no immediate output)").strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "...[truncated]"
        msg_lines.append(f"- {cmd_str}: {snippet}")
    msg_lines.append("")
    msg_lines.append("Hints:")
    msg_lines.append(" - Ensure distrobox/podman rootless runtime is available (enable user linger: sudo loginctl enable-linger <user>)")
    msg_lines.append(" - Ensure /etc/subuid and /etc/subgid contain mappings for the user and that /usr/bin/newuidmap and newgidmap are setuid root")
    msg = "\n".join(msg_lines)
    logger.error(msg)
    # record last failure for diagnostics and broadcast
    last_start_failure = msg
    broadcast_status_sync("error", {"message": msg, "current_model": None, "llama_server_running": False})
    return None


def rotate_llama_logs(current_log: Path, keep: int = 15):
    """Rotate llama-server logs, keeping the last N copies."""
    if not current_log.exists():
        return
    
    # Find existing rotated logs
    log_dir = current_log.parent
    base_name = current_log.stem
    suffix = current_log.suffix
    
    # Get all existing rotated logs sorted by number (descending)
    rotated_logs = []
    for f in log_dir.glob(f"{base_name}.*{suffix}"):
        try:
            num = int(f.stem.split(".")[-1])
            rotated_logs.append((num, f))
        except ValueError:
            continue
    
    rotated_logs.sort(key=lambda x: x[0], reverse=True)
    
    # Delete logs beyond the keep limit (accounting for the new rotation)
    for num, f in rotated_logs:
        if num >= keep:
            f.unlink()
            logger.debug(f"Deleted old llama-server log: {f}")
    
    # Rotate existing logs (N -> N+1)
    for num, f in rotated_logs:
        if num < keep:
            new_name = log_dir / f"{base_name}.{num + 1}{suffix}"
            f.rename(new_name)
    
    # Rotate current log to .1
    if current_log.exists():
        current_log.rename(log_dir / f"{base_name}.1{suffix}")


def stop_llama_server():
    """Stop the currently running llama-server."""
    global llama_process, llama_log_file, current_model
    
    server_config = config.get("server", {})
    distrobox_name = server_config.get("distrobox_name", "llama")
    
    # First, try to kill llama-server inside the distrobox
    try:
        subprocess.run(
            ["distrobox", "enter", distrobox_name, "--", "pkill", "-f", "llama-server"],
            timeout=10,
            capture_output=True
        )
        logger.info("Sent kill signal to llama-server inside distrobox")
    except Exception as e:
        logger.warning(f"Failed to kill llama-server inside distrobox: {e}")
    
    if llama_process is not None:
        logger.info(f"Stopping llama-server wrapper (PID: {llama_process.pid})")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server wrapper did not terminate gracefully, killing...")
            llama_process.kill()
            llama_process.wait()
        llama_process = None
        current_model = None
        logger.info("llama-server stopped")
    
    # Close log file if open
    if llama_log_file is not None and llama_log_file != subprocess.DEVNULL:
        try:
            llama_log_file.close()
        except Exception:
            pass
        llama_log_file = None


async def ensure_model_loaded(requested_model: Optional[str]) -> bool:
    """
    Ensure the requested model is loaded in llama-server.
    Returns True if the model is ready, False if there was an error.
    """
    global llama_process, current_model
    
    llama_model = get_local_model_name(requested_model)
    if llama_model is None:
        return False
    
    async with model_switch_lock:
        if current_model == llama_model and llama_process is not None:
            # Check if process is still running
            if llama_process.poll() is None:
                return True
            else:
                logger.warning("llama-server process died, restarting...")
        
        # Broadcast that we're switching models
        await broadcast_status("switching", {
            "target_model": llama_model,
            "previous_model": current_model
        })
        
        # Need to switch models or restart
        stop_llama_server()
        
        server_config = config.get("server", {})
        timeout = server_config.get("llama_startup_timeout", 300)
        
        llama_process = start_llama_server(llama_model)

        # If starting the process failed immediately (start_llama_server returns None),
        # fail fast instead of waiting the full timeout. start_llama_server already
        # broadcasts a detailed error message.
        if llama_process is None:
            logger.error(f"start_llama_server failed to spawn process for model {llama_model}")
            return False

        if await wait_for_llama_server(timeout):
            # Broadcast success
            await broadcast_status("ready", {
                "current_model": llama_model,
                "llama_server_running": True
            })
            return True
        else:
            # Broadcast failure
            await broadcast_status("error", {
                "message": f"Failed to load model {llama_model}",
                "current_model": None,
                "llama_server_running": False
            })
            stop_llama_server()
            return False


async def proxy_to_local(request: Request, path: str) -> Response:
    """Proxy request to local llama-server."""
    server_config = config.get("server", {})
    llama_port = server_config.get("llama_server_port", 8080)
    target_url = f"http://localhost:{llama_port}/{path}"
    
    # Get request body
    body = await request.body()

    # Log request
    log_request(request, body, "local")

    # Parse body once and determine method/key/model for attribution
    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}

    method = request.method.upper()
    key = f"{method} {request.url.path} -> local"
    # determine model for token attribution (fallback to current_model)
    model_name = None
    try:
        model_name = body_json.get('model')
    except Exception:
        model_name = None
    if not model_name:
        model_name = current_model

    # Token accounting: estimate tokens sent
    try:
        tokens_sent = 0
        # Chat-like payloads
        if isinstance(body_json, dict) and 'messages' in body_json:
            for m in body_json.get('messages', []):
                tokens_sent += count_text_tokens(str(m.get('content', '')), model_name)
        elif isinstance(body_json, dict) and 'input' in body_json:
            inp = body_json['input']
            if isinstance(inp, list):
                for it in inp:
                    tokens_sent += count_text_tokens(str(it), model_name)
            else:
                tokens_sent += count_text_tokens(str(inp), model_name)
        else:
            tokens_sent += count_text_tokens(body.decode('utf-8', errors='replace'), model_name)

        # schedule token increment
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_increment_tokens('sent', key, tokens_sent))
        except RuntimeError:
            asyncio.run(_increment_tokens('sent', key, tokens_sent))
    except Exception:
        pass
    
    # Forward headers (excluding host)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    # Check if streaming is requested
    is_streaming = body_json.get("stream", False)
    
    if is_streaming:
        # Streaming response - client must stay open during streaming
        client = httpx.AsyncClient(timeout=None)
        
        async def stream_generator():
            try:
                async with client.stream(
                    request.method,
                    target_url,
                    headers=headers,
                    content=body
                ) as response:
                    async for chunk in response.aiter_bytes():
                        # count tokens in this chunk (best-effort)
                        try:
                            chunk_text = chunk.decode('utf-8', errors='replace')
                            # simple heuristic: count tokens in chunk text
                            chunk_tokens = count_text_tokens(chunk_text, model_name)
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(_increment_tokens('recv', key, chunk_tokens))
                            except RuntimeError:
                                asyncio.run(_increment_tokens('recv', key, chunk_tokens))
                        except Exception:
                            pass
                        yield chunk
                        log_response_chunk(chunk)
            finally:
                await client.aclose()
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    else:
        # Non-streaming response
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.request(
                request.method,
                target_url,
                headers=headers,
                content=body
            )

            # Non-streaming: count tokens in response body
            try:
                resp_text = response.content.decode('utf-8', errors='replace')
                recv_tokens = count_text_tokens(resp_text, model_name)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_increment_tokens('recv', key, recv_tokens))
                except RuntimeError:
                    asyncio.run(_increment_tokens('recv', key, recv_tokens))
            except Exception:
                pass

            log_response(response.status_code, response.content)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )


async def proxy_to_remote(
    request: Request, 
    path: str, 
    model_config: dict
) -> Response:
    """Proxy request to remote API endpoint."""
    endpoint = model_config.get("endpoint", "")
    target_url = f"{endpoint}/{path}"
    
    # Get request body
    body = await request.body()
    
    # Log request
    log_request(request, body, "remote", endpoint)
    
    # Get API key
    api_key = None
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = model_config.get("api_key")
    
    # Forward headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    # Add API key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Add custom headers from config
    custom_headers = model_config.get("headers", {})
    headers.update(custom_headers)
    
    body_json = json.loads(body) if body else {}
    # Determine model name for attribution (may be provided in body)
    model_name = None
    try:
        model_name = body_json.get('model')
    except Exception:
        model_name = None
    if not model_name:
        model_name = current_model or model_config.get('name') or model_config.get('id') or 'unknown'

    is_streaming = body_json.get("stream", False)
    
    if is_streaming:
        # Streaming response - client must stay open during streaming
        client = httpx.AsyncClient(timeout=None)
        
        async def stream_generator():
            try:
                async with client.stream(
                    request.method,
                    target_url,
                    headers=headers,
                    content=body
                ) as response:
                    async for chunk in response.aiter_bytes():
                        # parse OpenAI-style SSE chunks for delta content when possible
                        try:
                            s = chunk.decode('utf-8', errors='replace')
                            # look for lines starting with 'data: '
                            texts = []
                            for line in s.splitlines():
                                line = line.strip()
                                if not line.startswith('data:'):
                                    continue
                                payload = line[5:].strip()
                                if payload == '[DONE]':
                                    continue
                                try:
                                    j = json.loads(payload)
                                    # extract any delta.content fields
                                    for choice in j.get('choices', []):
                                        delta = choice.get('delta', {})
                                        if isinstance(delta, dict) and 'content' in delta:
                                            texts.append(str(delta.get('content', '')))
                                except Exception:
                                    # fallback: treat payload as plain text
                                    texts.append(payload)
                            if texts:
                                chunk_text = '\n'.join(texts)
                                chunk_tokens = count_text_tokens(chunk_text, model_name)
                                try:
                                    loop = asyncio.get_running_loop()
                                    loop.create_task(_increment_tokens('recv', f"{request.method.upper()} {request.url.path} -> remote", chunk_tokens))
                                except RuntimeError:
                                    asyncio.run(_increment_tokens('recv', f"{request.method.upper()} {request.url.path} -> remote", chunk_tokens))
                        except Exception:
                            pass
                        yield chunk
                        log_response_chunk(chunk)
            finally:
                await client.aclose()
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    else:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.request(
                request.method,
                target_url,
                headers=headers,
                content=body
            )

            # Non-streaming: count tokens in response
            try:
                resp_text = response.content.decode('utf-8', errors='replace')
                # Use determined model_name (may be None)
                recv_tokens = count_text_tokens(resp_text, model_name)
                key = f"{request.method.upper()} {request.url.path} -> remote"
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_increment_tokens('recv', key, recv_tokens))
                except RuntimeError:
                    asyncio.run(_increment_tokens('recv', key, recv_tokens))
            except Exception:
                pass

            log_response(response.status_code, response.content)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )


def log_request(
    request: Request, 
    body: bytes, 
    target_type: str, 
    endpoint: str = "localhost"
):
    """Log incoming request."""
    try:
        body_str = body.decode("utf-8")[:2000] if body else ""
        logger.info(
            f"REQUEST [{target_type}] {request.method} {request.url.path} "
            f"-> {endpoint} | Body: {body_str}"
        )
    except Exception as e:
        logger.error(f"Error logging request: {e}")
    # Update request counts asynchronously
    try:
        path = request.url.path
        method = request.method.upper()

        # Determine model if available in body or current_model
        model_name = None
        try:
            body_json = json.loads(body) if body else {}
            model_name = body_json.get('model')
        except Exception:
            model_name = None

        if not model_name:
            model_name = current_model

        # Key: by endpoint only
        endpoint_key = f"{method} {path} -> {target_type}"

        # Increment endpoint key only
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_increment_count(endpoint_key))
        except RuntimeError:
            asyncio.run(_increment_count(endpoint_key))
    except Exception:
        pass


def log_response(status_code: int, content: bytes):
    """Log response."""
    try:
        content_str = content.decode("utf-8")[:2000] if content else ""
        logger.info(f"RESPONSE [{status_code}] | Body: {content_str}")
    except Exception as e:
        logger.error(f"Error logging response: {e}")


def log_response_chunk(chunk: bytes):
    """Log streaming response chunk."""
    try:
        chunk_str = chunk.decode("utf-8")[:500] if chunk else ""
        logger.debug(f"STREAM CHUNK | {chunk_str}")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, logger, llama_process
    
    # Startup
    config = load_config()
    logger = setup_logging(config)
    logger.info("Starting LLama Proxy Server")

    # One-time podman rootless state reset. After a reboot, crash-loop, or
    # when the service was previously run under incompatible systemd sandbox
    # settings, stale catatonit pause processes accumulate and leave podman
    # unable to create user namespaces ("invalid internal status",
    # "newuidmap: write to uid_map failed: Operation not permitted").
    #
    # The fix is to: (1) kill any stale pause processes, (2) run
    # `podman system migrate` to clean up internal state.
    # NOTE: this stops all running containers, so it must only run once here,
    # never inside the retry loop of start_llama_server.
    try:
        # Kill stale catatonit pause processes that accumulate during
        # crash-loop restarts and poison podman's namespace state.
        subprocess.run(
            ["pkill", "-9", "catatonit"],
            capture_output=True, timeout=5
        )
        time.sleep(1)  # let kernel clean up namespaces
        migrate_result = subprocess.run(
            ["podman", "system", "migrate"],
            capture_output=True, text=True, timeout=30
        )
        if migrate_result.returncode == 0:
            logger.info("podman system migrate completed successfully")
        else:
            logger.warning(
                f"podman system migrate returned {migrate_result.returncode}: "
                f"{migrate_result.stderr.strip()}"
            )
    except FileNotFoundError:
        logger.warning("podman not found in PATH, skipping system migrate")
    except Exception as e:
        logger.warning(f"podman system migrate failed: {e}")

    # Load the default model in a background task so uvicorn can finish
    # startup and begin accepting connections immediately. This avoids
    # blocking systemd (which waits for the lifespan to complete) for the
    # full model-load time (potentially 5+ minutes) and also handles the
    # boot-order race where distrobox/podman isn't ready yet.
    default_model = config.get("default_model", "qwen3")

    async def _load_default_model():
        """Load the default model with retries, running in the background."""
        max_attempts = 6
        retry_delays = [0, 30, 60, 120, 240, 300]  # first attempt immediate
        for attempt, delay in enumerate(retry_delays[:max_attempts], 1):
            if delay > 0:
                logger.info(
                    f"Background retry {attempt}/{max_attempts} for "
                    f"default model '{default_model}' in {delay}s"
                )
                await asyncio.sleep(delay)
            # If something else loaded a model while we were waiting, stop
            if current_model is not None:
                logger.info("Model already loaded by another request, stopping background loader")
                return
            try:
                if await ensure_model_loaded(default_model):
                    logger.info(f"Default model '{default_model}' loaded successfully")
                    return
            except Exception as e:
                logger.error(f"Exception loading default model (attempt {attempt}): {e}")
            logger.warning(f"Attempt {attempt}/{max_attempts} to load default model failed")
        logger.error(
            f"All attempts exhausted for default model '{default_model}'. "
            f"Model will be loaded on first matching request."
        )

    asyncio.get_running_loop().create_task(_load_default_model())

    # Load persisted request counts and token counts
    load_counts()
    load_token_counts()
    # Start background persist loops
    try:
        loop = asyncio.get_running_loop()
        global counts_persist_task, tokens_persist_task
        if counts_persist_task is None:
            counts_persist_task = loop.create_task(_counts_persist_loop())
        if tokens_persist_task is None:
            tokens_persist_task = loop.create_task(_tokens_persist_loop())
        global periodic_broadcast_task
        if periodic_broadcast_task is None:
            periodic_broadcast_task = loop.create_task(_periodic_broadcast_loop())
    except RuntimeError:
        pass
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLama Proxy Server")
    stop_llama_server()


app = FastAPI(
    title="LLama Proxy Server",
    description="Proxy server for routing OpenAI API requests",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the index page with API documentation."""
    # Build models table rows and quick link buttons for local models
    models_rows = ""
    model_buttons = ""
    model_options = ""
    for name, cfg in config.get("models", {}).items():
        model_type = cfg.get("type", "unknown")
        aliases = ", ".join(cfg.get("aliases", [])) or "—"
        endpoint = cfg.get("endpoint", "localhost:8080") if model_type == "remote" else "Local llama-server"
        type_badge = f'<span class="badge badge-local">Local</span>' if model_type == "local" else f'<span class="badge badge-remote">Remote</span>'
        
        # Build model dropdown options
        selected = "selected" if name == current_model else ""
        type_label = "Local" if model_type == "local" else "Remote"
        model_options += f'<option value="{name}" {selected}>{name} ({type_label})</option>'
        
        # Add switch button for local models that aren't currently loaded
        action_cell = ""
        if model_type == "local":
            llama_model = cfg.get("llama_model", name)
            if llama_model != current_model:
                action_cell = f'<button class="btn-switch" onclick="switchModel(\'{name}\')">Load Model</button>'
                model_buttons += f'<button class="btn-switch btn-model" onclick="switchModel(\'{name}\')">Load {name}</button>'
            else:
                action_cell = '<span class="badge badge-active">Active</span>'
        
        models_rows += f"""
        <tr>
            <td><code>{name}</code></td>
            <td>{type_badge}</td>
            <td><code>{aliases}</code></td>
            <td>{endpoint}</td>
            <td>{action_cell}</td>
        </tr>"""

    # Build list of local model names for JavaScript
    import json
    local_model_names = [name for name, cfg in config.get("models", {}).items() if cfg.get("type") == "local"]
    local_model_names_json = json.dumps(local_model_names)

    # Base URL from incoming request (includes scheme and host:port)
    base = str(request.base_url).rstrip('/')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLama Proxy Server</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #4f8cff;
            --accent-hover: #6ba1ff;
            --success: #4caf50;
            --warning: #ff9800;
            --border: #2a3a5a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; font-size: 1.1rem; }}
        .status-bar {{
            display: flex;
            gap: 2rem;
            background: var(--bg-card);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        .status-item {{ display: flex; align-items: center; gap: 0.5rem; }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .card {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        .card h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .card-header h2 {{
            margin-bottom: 0;
        }}
        .model-select-wrapper {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .endpoint-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .endpoint {{
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid var(--accent);
        }}
        .endpoint-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        .method {{
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .method-get {{ background: #2e7d32; color: #fff; }}
        .method-post {{ background: #1565c0; color: #fff; }}
        .endpoint-path {{ font-family: monospace; font-size: 0.95rem; color: var(--text-primary); }}
        .endpoint-desc {{ font-size: 0.85rem; color: var(--text-secondary); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-secondary); font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }}
        .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .badge-local {{ background: #2e7d32; color: #fff; }}
        .badge-remote {{ background: #7b1fa2; color: #fff; }}
        code {{
            background: var(--bg-primary);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        pre {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            margin-top: 1rem;
        }}
        .nav-links {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }}
        .nav-links a {{
            color: var(--accent);
            text-decoration: none;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 6px;
            border: 1px solid var(--border);
            transition: all 0.2s;
        }}
        .nav-links a:hover {{
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }}
        .btn-model {{
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
        .section-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .btn-switch {{
            background: var(--accent);
            color: #fff;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .btn-switch:hover {{
            background: var(--accent-hover);
        }}
        .btn-switch:disabled {{
            background: var(--text-secondary);
            cursor: not-allowed;
        }}
        .badge-active {{
            background: var(--success);
            color: #fff;
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .status-message {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            font-weight: 500;
            z-index: 1000;
            display: none;
        }}
        .status-message.success {{
            background: var(--success);
            color: #fff;
        }}
        .status-message.error {{
            background: #d32f2f;
            color: #fff;
        }}
        .status-message.loading {{
            background: var(--warning);
            color: #000;
        }}
        .quick-test {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }}
        .test-input-area, .test-output-area {{
            display: flex;
            flex-direction: column;
        }}
        .test-input-area label, .test-output-area label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .test-input {{
            width: 100%;
            min-height: 120px;
            padding: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
        }}
        .test-input:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .test-output {{
            width: 100%;
            min-height: 120px;
            max-height: 300px;
            padding: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: monospace;
            font-size: 0.85rem;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .test-hint {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}
        .test-status {{
            font-size: 0.8rem;
            margin-top: 0.5rem;
            color: var(--text-secondary);
        }}
        .test-status.streaming {{
            color: var(--success);
        }}
        .test-status.error {{
            color: #d32f2f;
        }}
        .btn-test {{
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 500;
            margin-left: auto;
            transition: all 0.2s;
        }}
        .btn-test:hover {{
            background: var(--accent);
            color: #fff;
        }}
        .api-test-section {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--accent);
        }}
        .api-test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            color: var(--accent);
        }}
        .model-select-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        .model-select {{
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.4rem 0.6rem;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .model-select:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .btn-close {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
            padding: 0 0.25rem;
        }}
        .btn-close:hover {{
            color: var(--text-primary);
        }}
        .api-test-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        .api-test-input-area, .api-test-output-area {{
            display: flex;
            flex-direction: column;
        }}
        .api-test-input-area label, .api-test-output-area label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .api-test-pre {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem;
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-primary);
            overflow: auto;
            max-height: 300px;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
        }}
        .stats-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        .stats-panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            font-weight: 500;
            color: var(--accent);
        }}
        .btn-close-stats {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.2rem;
            cursor: pointer;
            padding: 0 0.25rem;
            line-height: 1;
        }}
        .btn-close-stats:hover {{
            color: var(--text-primary);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px;
            background: var(--border);
        }}
        .stats-item {{
            display: flex;
            flex-direction: column;
            padding: 0.75rem 1rem;
            background: var(--bg-card);
        }}
        .stats-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }}
        .stats-value {{
            font-size: 0.95rem;
            color: var(--text-primary);
            font-family: monospace;
        }}
        .stats-unknown {{
            color: var(--warning);
            font-style: italic;
            cursor: help;
        }}
        .stats-toggle {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            color: var(--accent);
            cursor: pointer;
            margin-left: 1rem;
        }}
        .stats-toggle:hover {{
            background: var(--accent);
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLama Proxy Server</h1>
        <p class="subtitle">OpenAI-compatible API proxy for local and remote LLM models</p>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>Proxy Running</span>
            </div>
            <div class="status-item">
                <strong>Current Model:</strong>
                <code id="currentModelStatus">{current_model or 'None'}</code>
            </div>
            <div class="status-item">
                <strong>llama-server:</strong>
                <span id="llamaServerStatus">{'Running' if llama_process and llama_process.poll() is None else 'Stopped'}</span>
            </div>
        </div>

        <div id="statsPanel" class="stats-panel" style="display: none;">
            <div class="stats-panel-header">
                <span>Model Statistics</span>
                <button class="btn-close-stats" onclick="toggleStatsPanel()">&times;</button>
            </div>
            <div class="stats-grid">
                <div class="stats-item">
                    <span class="stats-label">Model</span>
                    <span class="stats-value" id="statsModel">-</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Llama-server status</span>
                    <span class="stats-value" id="statsLlamaStatus">-</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Max context</span>
                    <span class="stats-value" id="statsNCtx">
                        <span class="stats-val">-</span>
                        <span class="stats-unknown" title="Value not available from backend" style="display:none;">unknown</span>
                    </span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">KV cache tokens</span>
                    <span class="stats-value" id="statsKvCache">
                        <span class="stats-val">-</span>
                        <span class="stats-unknown" title="Value not available from backend" style="display:none;">unknown</span>
                    </span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Total tokens sent</span>
                    <span class="stats-value" id="statsTokensSent">0</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Total tokens received</span>
                    <span class="stats-value" id="statsTokensRecv">0</span>
                </div>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <p class="section-title" style="margin-bottom: 0;">Quick Links</p>
            <button class="stats-toggle" onclick="toggleStatsPanel()">Show Model Stats</button>
        </div>
        <div class="nav-links">
            <a href="/health">Health Check</a>
            <a href="/v1/models">List Models</a>
            <a href="/docs">OpenAPI Docs</a>
            <a href="/redoc">ReDoc</a>
            <a href="/logs">View Logs</a>
            {model_buttons}
        </div>

        <div class="card">
            <h2>Quick Test</h2>
            <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">
                Send a message to test the current model. Press Enter to send (Shift+Enter for new line).
            </p>
            <div class="quick-test">
                <div class="test-input-area">
                    <label>Input</label>
                    <textarea id="testInput" class="test-input" placeholder="Type your message here..."></textarea>
                    <p class="test-hint">Press Enter to send, Shift+Enter for new line</p>
                </div>
                <div class="test-output-area">
                    <label>Response</label>
                    <div id="testOutput" class="test-output"></div>
                    <p id="testStatus" class="test-status"></p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Configured Models</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model ID</th>
                        <th>Type</th>
                        <th>Aliases (supports wildcards)</th>
                        <th>Endpoint</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {models_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>API Passthrough Endpoints</h2>
                <div class="model-select-wrapper">
                    <label for="modelSelect" class="model-select-label">Test with model:</label>
                    <select id="modelSelect" class="model-select" onchange="updateTestRequest()">
                        {model_options}
                    </select>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                These endpoints are fully compatible with the OpenAI API. Requests are automatically routed to local llama-server or remote APIs based on the model specified. Click "Test" to try each endpoint.
            </p>
            <div class="endpoint-grid">
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/chat/completions</span>
                        <button class="btn-test" onclick="testEndpoint('chat')">Test</button>
                    </div>
                    <p class="endpoint-desc">Chat completions - send messages and get AI responses</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/completions</span>
                        <button class="btn-test" onclick="testEndpoint('completions')">Test</button>
                    </div>
                    <p class="endpoint-desc">Text completions - complete a prompt</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-get">GET</span>
                        <span class="endpoint-path">/v1/models</span>
                        <button class="btn-test" onclick="testEndpoint('models')">Test</button>
                    </div>
                    <p class="endpoint-desc">List all available models</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/v1/embeddings</span>
                        <button class="btn-test" onclick="testEndpoint('embeddings')">Test</button>
                    </div>
                    <p class="endpoint-desc">Generate embeddings for text</p>
                </div>
            </div>
            
            <div id="apiTestSection" class="api-test-section" style="display: none; margin-top: 1.5rem;">
                <div class="api-test-header">
                    <strong id="apiTestTitle">Test Request</strong>
                    <button class="btn-close" onclick="closeApiTest()">&times;</button>
                </div>
                <div class="api-test-grid">
                    <div class="api-test-input-area">
                        <label>Request</label>
                        <pre id="apiTestRequest" class="api-test-pre"></pre>
                    </div>
                    <div class="api-test-output-area">
                        <label>Response</label>
                        <pre id="apiTestResponse" class="api-test-pre"></pre>
                        <p id="apiTestStatus" class="test-status"></p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Admin Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-get">GET</span>
                        <span class="endpoint-path">/health</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="refreshStatus()">Refresh</button>
                    </div>
                    <p class="endpoint-desc">Health check - returns server and model status</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/reload-config</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="reloadConfig()">Reload</button>
                    </div>
                    <p class="endpoint-desc">Reload configuration from config.yaml</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/switch-model/{{model}}</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="adminSwitchModel()">Switch To Selected</button>
                    </div>
                    <p class="endpoint-desc">Switch the llama-server to the model selected in the dropdown above</p>
                </div>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method method-post">POST</span>
                        <span class="endpoint-path">/admin/stop-server</span>
                        <button class="btn-switch" style="margin-left:auto; background:#d32f2f;" onclick="stopServer()">Stop</button>
                    </div>
                    <p class="endpoint-desc">Stop the llama-server process (requires confirmation)</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Model Routing</h2>
            <p style="color: var(--text-secondary);">
                The proxy automatically routes requests based on the <code>model</code> parameter in your API request:
            </p>
            <ul style="margin: 1rem 0 0 1.5rem; color: var(--text-secondary);">
                <li><strong>Local models</strong> are served by llama-server running in a distrobox container</li>
                <li><strong>Remote models</strong> are proxied to external APIs (OpenAI, Anthropic, etc.)</li>
                <li><strong>Wildcard aliases</strong> like <code>gpt-*</code> match any model starting with that prefix</li>
                <li>If a model switch is needed, the server will automatically load the new model</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>API Endpoints</h2>
            <pre style="background:var(--bg-primary); padding:1rem; border-radius:6px; color:var(--text-secondary);">==========================================
API Endpoints
==========================================

  Health check:     GET  {base}/health
  List models:      GET  {base}/v1/models
  Chat completions: POST {base}/v1/chat/completions
  Completions:      POST {base}/v1/completions

  Admin endpoints:
    Reload config:  POST {base}/admin/reload-config
    Switch model:   POST {base}/admin/switch-model/{{model}}
    Stop server:    POST {base}/admin/stop-server
</pre>
        </div>
    </div>

    <div id="statusMessage" class="status-message"></div>

    <script>
        async function switchModel(modelName) {{
            const statusEl = document.getElementById('statusMessage');
            const currentModelEl = document.getElementById('currentModelStatus');
            const llamaStatusEl = document.getElementById('llamaServerStatus');
            const btn = event.target;
            
            // Store original values for error recovery
            const originalModel = currentModelEl.textContent;
            const originalLlamaStatus = llamaStatusEl.textContent;
            
            // Show loading state - update status bar immediately
            btn.disabled = true;
            btn.textContent = 'Loading...';
            currentModelEl.textContent = `Switching to ${{modelName}}...`;
            llamaStatusEl.textContent = 'Switching';
            statusEl.className = 'status-message loading';
            statusEl.textContent = `Switching model to ${{modelName}}... This may take a few minutes.`;
            statusEl.style.display = 'block';
            
            try {{
                const response = await fetch(`/admin/switch-model/${{modelName}}`, {{
                    method: 'POST'
                }});
                
                const data = await response.json();
                
                if (response.ok) {{
                    // Update status bar with new model
                    currentModelEl.textContent = modelName;
                    llamaStatusEl.textContent = 'Running';
                    statusEl.className = 'status-message success';
                    statusEl.textContent = `Successfully switched to ${{modelName}}`;
                    // Reload page after short delay to update UI
                    setTimeout(() => window.location.reload(), 1500);
                }} else {{
                    throw new Error(data.detail || 'Failed to switch model');
                }}
            }} catch (error) {{
                // Restore original values on error
                currentModelEl.textContent = originalModel;
                llamaStatusEl.textContent = originalLlamaStatus;
                statusEl.className = 'status-message error';
                statusEl.textContent = `Error: ${{error.message}}`;
                btn.disabled = false;
                btn.textContent = 'Load Model';
                // Hide error after 5 seconds
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        // Status bar elements
        const currentModelEl = document.getElementById('currentModelStatus');
        const llamaStatusEl = document.getElementById('llamaServerStatus');
        const statusEl = document.getElementById('statusMessage');
        
        // Stats panel elements
        const statsPanel = document.getElementById('statsPanel');
        const statsModelEl = document.getElementById('statsModel');
        const statsLlamaStatusEl = document.getElementById('statsLlamaStatus');
        const statsNCtxEl = document.getElementById('statsNCtx');
        const statsKvCacheEl = document.getElementById('statsKvCache');
        const statsTokensSentEl = document.getElementById('statsTokensSent');
        const statsTokensRecvEl = document.getElementById('statsTokensRecv');
        
        // Track the actual current model (updated after successful operations)
        let actualCurrentModel = '{current_model or "None"}';
        
        // Toggle stats panel visibility
        function toggleStatsPanel() {{
            if (statsPanel.style.display === 'none') {{
                statsPanel.style.display = 'block';
                document.querySelector('.stats-toggle').textContent = 'Hide Model Stats';
            }} else {{
                statsPanel.style.display = 'none';
                document.querySelector('.stats-toggle').textContent = 'Show Model Stats';
            }}
        }}
        
        // Update stats panel with new values (only if changed)
        function updateStatsPanel(data) {{
            if (!data) return;
            
            if (data.current_model !== undefined) {{
                const val = data.current_model || '-';
                if (statsModelEl.textContent !== val) {{
                    statsModelEl.textContent = val;
                }}
            }}
            
            if (data.llama_server_running !== undefined) {{
                const val = data.llama_server_running ? 'Running' : 'Stopped';
                if (statsLlamaStatusEl.textContent !== val) {{
                    statsLlamaStatusEl.textContent = val;
                }}
            }}
            
            if (data.n_ctx !== undefined) {{
                const valSpan = statsNCtxEl.querySelector('.stats-val');
                const unknownSpan = statsNCtxEl.querySelector('.stats-unknown');
                if (data.n_ctx === null || data.n_ctx === undefined) {{
                    if (valSpan) valSpan.textContent = '-';
                    if (unknownSpan) unknownSpan.style.display = 'inline';
                }} else {{
                    if (valSpan) valSpan.textContent = data.n_ctx;
                    if (unknownSpan) unknownSpan.style.display = 'none';
                }}
            }}
            
            if (data.kv_cache_tokens !== undefined) {{
                const valSpan = statsKvCacheEl.querySelector('.stats-val');
                const unknownSpan = statsKvCacheEl.querySelector('.stats-unknown');
                if (data.kv_cache_tokens === null || data.kv_cache_tokens === undefined) {{
                    if (valSpan) valSpan.textContent = '-';
                    if (unknownSpan) unknownSpan.style.display = 'inline';
                }} else {{
                    if (valSpan) valSpan.textContent = data.kv_cache_tokens;
                    if (unknownSpan) unknownSpan.style.display = 'none';
                }}
            }}
            
            if (data.total_sent !== undefined) {{
                const val = String(data.total_sent);
                if (statsTokensSentEl.textContent !== val) {{
                    statsTokensSentEl.textContent = val;
                }}
            }}
            
            if (data.total_recv !== undefined) {{
                const val = String(data.total_recv);
                if (statsTokensRecvEl.textContent !== val) {{
                    statsTokensRecvEl.textContent = val;
                }}
            }}
        }}
        
        // Helper function to show model switching status
        function showSwitchingStatus(targetModel) {{
            currentModelEl.textContent = `Switching to ${{targetModel}}...`;
            llamaStatusEl.textContent = 'Switching';
            statusEl.className = 'status-message loading';
            statusEl.textContent = `Switching model to ${{targetModel}}... This may take a few minutes.`;
            statusEl.style.display = 'block';
        }}
        
        // Helper function to update status after successful model load
        function showModelReady(modelName) {{
            actualCurrentModel = modelName;
            currentModelEl.textContent = modelName;
            llamaStatusEl.textContent = 'Running';
            statusEl.className = 'status-message success';
            statusEl.textContent = `Model ${{modelName}} is ready`;
            // Hide success message after 3 seconds
            setTimeout(() => statusEl.style.display = 'none', 3000);
        }}
        
        // Helper function to check if model switch is needed and show status
        function checkAndShowSwitchStatus(targetModel) {{
            // Check if this is a local model that might need switching
            const localModels = {local_model_names_json};
            const isLocal = localModels.some(m => targetModel.toLowerCase().startsWith(m.toLowerCase()));
            
            if (isLocal && targetModel !== actualCurrentModel) {{
                showSwitchingStatus(targetModel);
                return true;
            }}
            return false;
        }}
        
        // Helper to refresh status from server
        async function refreshStatus() {{
            try {{
                const response = await fetch('/health');
                const data = await response.json();
                if (data.current_model) {{
                    actualCurrentModel = data.current_model;
                    currentModelEl.textContent = data.current_model;
                    llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                }}
            }} catch (e) {{
                // Ignore errors
            }}
        }}

        // Subscribe to Server-Sent Events for real-time status updates
        function connectSSE() {{
            const eventSource = new EventSource('/events');
            
            eventSource.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    
                    switch (data.type) {{
                        case 'status':
                            if (data.current_model) {{
                                actualCurrentModel = data.current_model;
                                currentModelEl.textContent = data.current_model;
                            }}
                            llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                            updateStatsPanel(data);
                            break;
                            
                        case 'switching':
                            // Model switch started
                            showSwitchingStatus(data.target_model);
                            break;
                            
                        case 'ready':
                            // Model switch completed successfully
                            showModelReady(data.current_model);
                            break;
                            
                        case 'error':
                            // Model switch failed
                            currentModelEl.textContent = data.current_model || 'None';
                            llamaStatusEl.textContent = data.llama_server_running ? 'Running' : 'Stopped';
                            statusEl.className = 'status-message error';
                            statusEl.textContent = data.message || 'An error occurred';
                            statusEl.style.display = 'block';
                            setTimeout(() => statusEl.style.display = 'none', 5000);
                            break;
                    }}
                }} catch (e) {{
                    console.error('Error parsing SSE message:', e);
                }}
            }};
            
            eventSource.onerror = () => {{
                // Reconnect after a delay
                eventSource.close();
                setTimeout(connectSSE, 5000);
            }};
        }}
        
        // Start SSE connection
        connectSSE();

        // Admin button handlers
        async function reloadConfig() {{
            try {{
                const resp = await fetch('/admin/reload-config', {{ method: 'POST' }});
                const data = await resp.json();
                statusEl.className = 'status-message success';
                statusEl.textContent = data.message || 'Config reloaded';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 3000);
                await refreshStatus();
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = 'Failed to reload config';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        async function adminSwitchModel() {{
            const selected = modelSelect ? modelSelect.value : null;
            if (!selected) return;
            try {{
                showSwitchingStatus(selected);
                const resp = await fetch(`/admin/switch-model/${{selected}}`, {{ method: 'POST' }});
                const data = await resp.json();
                if (resp.ok) {{
                    statusEl.className = 'status-message success';
                    statusEl.textContent = data.message || `Switched to ${{selected}}`;
                    statusEl.style.display = 'block';
                    setTimeout(() => statusEl.style.display = 'none', 3000);
                    await refreshStatus();
                    window.location.reload();
                }} else {{
                    throw new Error(data.detail || 'Switch failed');
                }}
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = `Error: ${{e.message}}`;
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
                await refreshStatus();
            }}
        }}

        async function stopServer() {{
            if (!confirm('Are you sure you want to stop the llama-server?')) return;
            try {{
                const resp = await fetch('/admin/stop-server', {{ method: 'POST' }});
                const data = await resp.json();
                statusEl.className = 'status-message success';
                statusEl.textContent = data.message || 'llama-server stopped';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 3000);
                await refreshStatus();
            }} catch (e) {{
                statusEl.className = 'status-message error';
                statusEl.textContent = 'Failed to stop server';
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }}
        }}

        // Quick Test functionality
        const testInput = document.getElementById('testInput');
        const testOutput = document.getElementById('testOutput');
        const testStatus = document.getElementById('testStatus');
        let isStreaming = false;

        testInput.addEventListener('keydown', async (e) => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                if (isStreaming) return;
                
                const message = testInput.value.trim();
                if (!message) return;
                
                await sendTestMessage(message);
            }}
        }});

        async function sendTestMessage(message) {{
            isStreaming = true;
            testOutput.textContent = '';
            testStatus.textContent = 'Connecting...';
            testStatus.className = 'test-status';
            
            try {{
                const response = await fetch('/v1/chat/completions', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        model: actualCurrentModel,
                        messages: [{{ role: 'user', content: message }}],
                        stream: true
                    }})
                }});
                
                if (!response.ok) {{
                    const err = await response.json();
                    throw new Error(err.detail || `HTTP ${{response.status}}`);
                }}
                
                testStatus.textContent = 'Streaming...';
                testStatus.className = 'test-status streaming';
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {{
                                const json = JSON.parse(data);
                                const content = json.choices?.[0]?.delta?.content;
                                if (content) {{
                                    testOutput.textContent += content;
                                    testOutput.scrollTop = testOutput.scrollHeight;
                                }}
                            }} catch {{
                                // Skip invalid JSON
                            }}
                        }}
                    }}
                }}
                
                testStatus.textContent = 'Complete';
                testStatus.className = 'test-status';
            }} catch (error) {{
                testStatus.textContent = `Error: ${{error.message}}`;
                testStatus.className = 'test-status error';
            }} finally {{
                isStreaming = false;
                // Refresh status bar in case model changed
                await refreshStatus();
            }}
        }}

        // API Endpoint Test functionality
        const apiTestSection = document.getElementById('apiTestSection');
        const apiTestTitle = document.getElementById('apiTestTitle');
        const apiTestRequest = document.getElementById('apiTestRequest');
        const apiTestResponse = document.getElementById('apiTestResponse');
        const apiTestStatus = document.getElementById('apiTestStatus');
        const modelSelect = document.getElementById('modelSelect');
        
        let currentEndpointType = null;

        function getTestExample(endpointType) {{
            const selectedModel = modelSelect ? modelSelect.value : '{current_model or "qwen3"}';
            
            const examples = {{
                chat: {{
                    title: 'POST /v1/chat/completions',
                    method: 'POST',
                    url: '/v1/chat/completions',
                    body: {{
                        model: selectedModel,
                        messages: [{{ role: 'user', content: 'Say hello in exactly 3 words.' }}],
                        max_tokens: 50
                    }}
                }},
                completions: {{
                    title: 'POST /v1/completions',
                    method: 'POST',
                    url: '/v1/completions',
                    body: {{
                        model: selectedModel,
                        prompt: 'The quick brown fox',
                        max_tokens: 30
                    }}
                }},
                models: {{
                    title: 'GET /v1/models',
                    method: 'GET',
                    url: '/v1/models',
                    body: null
                }},
                embeddings: {{
                    title: 'POST /v1/embeddings',
                    method: 'POST',
                    url: '/v1/embeddings',
                    body: {{
                        model: selectedModel,
                        input: 'Hello, world!'
                    }}
                }}
            }};
            
            return examples[endpointType];
        }}
        
        function updateTestRequest() {{
            if (!currentEndpointType) return;
            
            const example = getTestExample(currentEndpointType);
            if (example && example.body) {{
                apiTestRequest.textContent = JSON.stringify(example.body, null, 2);
            }}
        }}

        async function testEndpoint(endpointType) {{
            currentEndpointType = endpointType;
            const example = getTestExample(endpointType);
            if (!example) return;

            // Check if we need to show model switching status
            const selectedModel = modelSelect ? modelSelect.value : actualCurrentModel;
            const willSwitch = checkAndShowSwitchStatus(selectedModel);

            // Show the test section
            apiTestSection.style.display = 'block';
            apiTestTitle.textContent = example.title;
            
            // Format and display the request
            if (example.body) {{
                apiTestRequest.textContent = JSON.stringify(example.body, null, 2);
            }} else {{
                apiTestRequest.textContent = '(No request body - GET request)';
            }}
            
            apiTestResponse.textContent = '';
            apiTestStatus.textContent = willSwitch ? 'Switching model...' : 'Sending request...';
            apiTestStatus.className = 'test-status';

            // Scroll to the test section
            apiTestSection.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});

            try {{
                const fetchOptions = {{
                    method: example.method,
                    headers: {{}}
                }};

                if (example.body) {{
                    fetchOptions.headers['Content-Type'] = 'application/json';
                    fetchOptions.body = JSON.stringify(example.body);
                }}

                const response = await fetch(example.url, fetchOptions);
                // Prefer JSON parsing, but gracefully handle non-JSON responses (HTML/text)
                let data;
                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {{
                    try {{
                        data = await response.json();
                    }} catch (e) {{
                        // Malformed JSON despite content-type; fall back to text
                        data = await response.text();
                    }}
                }} else {{
                    // Not JSON - try to parse as JSON, otherwise keep as plain text
                    const txt = await response.text();
                    try {{
                        data = JSON.parse(txt);
                    }} catch (e) {{
                        data = txt;
                    }}
                }}

                // Update status bar after request completes (model may have switched)
                await refreshStatus();

                const formatted = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

                if (response.ok) {{
                    apiTestResponse.textContent = formatted;
                    apiTestStatus.textContent = `Success (HTTP ${{response.status}})`;
                    apiTestStatus.className = 'test-status streaming';
                }} else {{
                    apiTestResponse.textContent = formatted;
                    apiTestStatus.textContent = `Error (HTTP ${{response.status}})`;
                    apiTestStatus.className = 'test-status error';
                }}
            }} catch (error) {{
                apiTestResponse.textContent = error.message;
                apiTestStatus.textContent = 'Request failed';
                apiTestStatus.className = 'test-status error';
                // Still try to refresh status on error
                await refreshStatus();
            }}
        }}

        function closeApiTest() {{
            apiTestSection.style.display = 'none';
            currentEndpointType = null;
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "current_model": current_model,
        "llama_server_running": llama_process is not None and llama_process.poll() is None
    }


@app.get("/events")
async def status_events():
    """Server-Sent Events endpoint for real-time status updates."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    sse_clients.add(queue)

    async def event_generator():
        try:
            llama_status = await query_llama_status()
            total_sent = token_counts.get("total_sent", 0)
            total_recv = token_counts.get("total_recv", 0)
            
            initial_status = json.dumps({
                "type": "status",
                "current_model": current_model,
                "llama_server_running": llama_status["llama_server_running"],
                "n_ctx": llama_status["n_ctx"],
                "kv_cache_tokens": llama_status["kv_cache_tokens"],
                "total_sent": total_sent,
                "total_recv": total_recv
            })
            yield f"data: {initial_status}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # keepalive comment
                    yield ": keepalive\n\n"
        finally:
            sse_clients.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/v1/models")
async def list_models():
    """List available models from proxy configuration."""
    models_list = []
    
    for name, cfg in config.get("models", {}).items():
        models_list.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local" if cfg.get("type") == "local" else "remote",
            "type": cfg.get("type"),
            "aliases": cfg.get("aliases", [])
        })
    
    return {
        "object": "list",
        "data": models_list
    }


@app.get("/logs/tail")
async def tail_logs(request: Request, lines: int = 100):
    """Stream the proxy log file as Server-Sent Events (SSE).

    Query params:
    - lines: number of previous lines to include initially (default 100)

    Sends an initial SSE message with key `initial` containing the last
    `lines` lines, then streams new lines as they are appended with key
    `line`.
    """
    # Resolve log file path
    if log_dir:
        log_path = log_dir / "proxy.log"
    else:
        log_path = Path(__file__).parent / "logs" / "proxy.log"

    async def event_generator():
        # local reference to counts queue for cleanup in finally - ensure always defined
        _local_counts_queue = None

        try:
            if not log_path.exists():
                err = {"error": "log_not_found", "path": str(log_path)}
                yield f"data: {json.dumps(err)}\n\n"
                return

            # Helper to read last N lines in a thread
            def read_last_n(n: int) -> str:
                # Read in binary for efficient seeking
                with open(log_path, "rb") as f:
                    f.seek(0, 2)
                    filesize = f.tell()
                    block_size = 1024
                    data = b""
                    # Read backwards until we have enough lines or hit BOF
                    while filesize > 0 and data.count(b"\n") <= n:
                        read_size = min(block_size, filesize)
                        f.seek(filesize - read_size)
                        chunk = f.read(read_size)
                        data = chunk + data
                        filesize -= read_size
                    lines_bytes = data.splitlines()[-n:]
                    return b"\n".join(lines_bytes).decode("utf-8", errors="replace")

            # Send initial block of lines as an array of {line, severity} objects
            initial = await asyncio.to_thread(read_last_n, lines)
            initial_lines = []
            try:
                for l in initial.splitlines():
                    try:
                        sev = detect_log_severity(l)
                    except Exception:
                        sev = None
                    initial_lines.append({'line': l, 'severity': sev})
            except Exception:
                # Fallback: send the raw initial string if something goes wrong
                yield f"data: {json.dumps({'initial': initial})}\n\n"
            else:
                yield f"data: {json.dumps({'initial': initial_lines})}\n\n"

            # Register for counts updates
            counts_queue: asyncio.Queue | None = None
            try:
                counts_queue = asyncio.Queue(maxsize=10)
                log_tail_clients.add(counts_queue)
            except Exception:
                counts_queue = None

            # Start following the file
            last_pos = log_path.stat().st_size
            # local reference to the counts queue
            _local_counts_queue = counts_queue if counts_queue is not None else None

            while True:
                # If client disconnected, stop
                if await asyncio.sleep(0):
                    pass

                # Small sleep / wait for counts updates to avoid busy loop
                try:
                    # Wait briefly for any counts/tokens updates to arrive on the queue.
                    update = None
                    if _local_counts_queue is not None:
                        try:
                            update = await asyncio.wait_for(_local_counts_queue.get(), timeout=0.25)
                        except asyncio.TimeoutError:
                            update = None
                    else:
                        await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    break

                # If we got an update, send it immediately and continue (don't wait for file checks)
                if update is not None:
                    try:
                        yield f"data: {json.dumps(update)}\n\n"
                    except Exception:
                        pass
                    continue
                # If file was rotated/recreated, reset position
                try:
                    cur_stat = log_path.stat()
                except FileNotFoundError:
                    # File disappeared; notify and exit
                    yield f"data: {json.dumps({'info': 'log_rotated_or_removed'})}\n\n"
                    break

                cur_size = cur_stat.st_size
                if cur_size < last_pos:
                    # File truncated/rotated
                    last_pos = 0

                if cur_size > last_pos:
                    # Read new data
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_pos)
                        new = f.read()
                    last_pos = cur_size

                    # Send each new line as its own SSE message
                    for line in new.splitlines():
                        try:
                            sev = detect_log_severity(line)
                        except Exception:
                            sev = None
                        yield f"data: {json.dumps({'line': line, 'severity': sev})}\n\n"
                else:
                    # No new file data; send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Cleanup
            try:
                if _local_counts_queue is not None:
                    log_tail_clients.discard(_local_counts_queue)
            except Exception:
                pass
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/logs")
async def view_logs(request: Request):
    """Simple web UI to view the tailed proxy log using SSE from /logs/tail."""
    base = str(request.base_url).rstrip('/')
    # Prepare a summary HTML fragment for counts
    async def get_counts_html():
        # Build a stable sorted view
        items = []
        async with counts_lock:
            for k, v in request_counts.items():
                items.append((k, v))
        items.sort(key=lambda x: (-x[1], x[0]))
        rows = '\n'.join([f'<div class="line">{k}: <strong>{v}</strong></div>' for k, v in items])
        if not rows:
            rows = '<div class="muted">No requests recorded yet.</div>'
        return rows

    counts_html = await get_counts_html()
    # Prepare tokens HTML fragment
    async def get_tokens_html():
        items = []
        async with token_lock:
            for k, v in token_counts.items():
                items.append((k, v))
        # show totals first
        totals = []
        for k, v in items:
            if k.startswith('total_'):
                totals.append((k, v))
        other = [(k, v) for k, v in items if not k.startswith('total_')]
        other.sort(key=lambda x: (-x[1], x[0]))
        rows = ''
        for k, v in totals:
            rows += f'<div class="line">{k}: <strong>{v}</strong></div>'
        for k, v in other:
            rows += f'<div class="line">{k}: <strong>{v}</strong></div>'
        if not rows:
            rows = '<div class="muted">No token stats yet.</div>'
        return rows

    tokens_html = await get_tokens_html()

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proxy Log Tail</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #4f8cff;
            --accent-hover: #6ba1ff;
            --success: #4caf50;
            --warning: #ff9800;
            --border: #2a3a5a;
            /* Severity colours default to existing tokens when available */
            --log-error: var(--error, #ff4d4f);
            --log-warning: var(--warning, #ff9800);
            --log-info: var(--accent, #4f8cff);
            --log-debug: var(--text-secondary, #9aa4b2);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 1rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display:flex; gap:1rem; align-items:center; margin-bottom:1rem; }
        .controls { display:flex; gap:0.5rem; align-items:center; }
        input[type=number] { width:6rem; padding:0.35rem; border-radius:6px; border:1px solid var(--border); background:var(--bg-card); color:var(--text-primary); }
        button { background:var(--accent); color:#fff; border:none; padding:0.4rem 0.6rem; border-radius:6px; cursor:pointer; }
        button:disabled { opacity:0.5; cursor:not-allowed; }
        .log { height: calc(100vh - 140px); overflow:auto; padding:1rem; font-family: monospace; background: linear-gradient(180deg, var(--bg-card), rgba(15,18,30,1)); border-radius:8px; border:1px solid var(--border); white-space:pre-wrap; }
        .line { display:flex; gap:0.5rem; align-items:center; padding:0.25rem 0; border-bottom:1px solid rgba(255,255,255,0.02); }
        .muted { color:var(--text-secondary); font-size:0.9rem; }

        /* Severity badge */
        .badge-sev {
            display:inline-flex; align-items:center; gap:0.4rem; padding:0.12rem 0.45rem; border-radius:6px; font-weight:700; font-size:0.75rem; color:#fff;
        }
        .sev-error { background: var(--log-error); }
        .sev-warning { background: var(--log-warning); }
        .sev-info { background: var(--log-info); }
        .sev-debug { background: var(--log-debug); color: #000; }

        .sev-icon { font-weight:700; margin-right:0.2rem; }
        .line-text { white-space:pre-wrap; color:var(--text-primary); }

        /* When the user enables colourization, apply semantic colours to the
           line text itself for better scanability. The outer `.colorize-on`
           class is toggled by the Colorize checkbox so users can disable this
           behaviour without losing the badge/text markers. */
        .colorize-on .line.sev-error .line-text { color: var(--log-error); }
        .colorize-on .line.sev-warning .line-text { color: var(--log-warning); }
        .colorize-on .line.sev-info .line-text { color: var(--log-info); }
        .colorize-on .line.sev-debug .line-text { color: var(--log-debug); }

        /* Low-contrast fallback: ensure badges have minimum contrast */
        @media (prefers-contrast: high) {
          .badge-sev { box-shadow: 0 0 0 2px rgba(0,0,0,0.15) inset; }
        }
    </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div style="display:flex; align-items:center; gap:1rem; margin-right:1rem;">
        <a href="/" style="color:var(--accent); text-decoration:none; font-weight:600;">Home</a>
      </div>
      <div class="controls">
        <label class="muted">Lines:</label>
        <input id="lines" type="number" value="200" min="1" />
        <button id="connect">Connect</button>
        <button id="disconnect" disabled>Disconnect</button>
        <button id="clear">Clear</button>
        <label class="muted" style="margin-left:12px;"><input id="autoscroll" type="checkbox" checked /> Auto-scroll</label>
        <label class="muted" style="margin-left:8px;"><input id="colorize" type="checkbox" checked /> Colorize</label>
        <button id="download">Download</button>
      </div>
    </div>
    <div style="margin-bottom:0.75rem;">
      <h3 style="margin:0 0 0.5rem 0; color:var(--accent);">Request Summary</h3>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
        <div style="background:var(--bg-card); padding:0.75rem; border-radius:6px; border:1px solid var(--border); max-height:160px; overflow:auto;">
          <h4 class="muted">Counts</h4>
          <div id="counts"></div>
          <pre id="rawCounts" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
        <div style="background:var(--bg-card); padding:0.75rem; border-radius:6px; border:1px solid var(--border); max-height:160px; overflow:auto;">
          <h4 class="muted">Tokens</h4>
          <div id="tokens"></div>
          <pre id="rawTokens" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
      </div>
    </div>
    <div id="log" class="log" role="log" aria-live="polite"></div>
  </div>

  <script>
    const logEl = document.getElementById('log');
    const linesInput = document.getElementById('lines');
    const connectBtn = document.getElementById('connect');
    const disconnectBtn = document.getElementById('disconnect');
    const clearBtn = document.getElementById('clear');
    const autoscrollCb = document.getElementById('autoscroll');
    const colorizeCb = document.getElementById('colorize');
    const downloadBtn = document.getElementById('download');
    let es = null;

    // Heuristics to detect severity from a log line.
    // Improved to find common severity tokens anywhere in the line (timestamps
    // and prefixes like "2026-... - INFO - ..." are handled).
    function detectSeverity(line) {
      try {
        const trimmed = (line || '').toString().trim();

        // 1) If the line is structured JSON and contains a standard level field,
        //    prefer that value (common keys: level, severity, levelname).
        if (trimmed.startsWith('{')) {
          try {
            const j = JSON.parse(trimmed);
            const lvlRaw = j.level || j.severity || j.levelname || j.level_name || j.levelName || j.loglevel;
            if (lvlRaw) {
              const lvl = String(lvlRaw).toLowerCase();
              if (lvl.indexOf('err') !== -1) return 'error';
              if (lvl.indexOf('warn') !== -1) return 'warning';
              if (lvl.indexOf('info') !== -1) return 'info';
              if (lvl.indexOf('debug') !== -1 || lvl.indexOf('trace') !== -1 || lvl.indexOf('verb') !== -1) return 'debug';
            }
          } catch (e) {
            // not JSON - continue to text heuristics
          }
        }

        // 2) Text heuristics: split on non-alpha characters and look for tokens.
        //    Splitting is more robust than relying on \b word boundaries which
        //    can be affected by punctuation or unusual unicode characters.
        const up = (trimmed || '').toUpperCase();
        const parts = up.split(/[^A-Z]+/);
        for (let i = 0; i < parts.length; i++) {
          const p = parts[i];
          if (!p) continue;
          if (p === 'ERROR' || p === 'ERR') return 'error';
          if (p === 'WARN' || p === 'WARNING') return 'warning';
          if (p === 'INFO') return 'info';
          if (p === 'DEBUG' || p === 'TRACE' || p === 'VERB') return 'debug';
        }

        // 3) Fallback substring checks for common log patterns
        if (up.indexOf(' - INFO - ') !== -1 || up.indexOf('[INFO]') !== -1 || /LEVEL[:=]\s*INFO/.test(up)) return 'info';
        if (up.indexOf(' - ERROR - ') !== -1 || up.indexOf('[ERROR]') !== -1 || /LEVEL[:=]\s*ERROR/.test(up)) return 'error';
        if (up.indexOf(' - WARN - ') !== -1 || up.indexOf('[WARN]') !== -1 || /LEVEL[:=]\s*WARN/.test(up)) return 'warning';
        if (up.indexOf(' - DEBUG - ') !== -1 || up.indexOf('[DEBUG]') !== -1 || /LEVEL[:=]\s*DEBUG/.test(up)) return 'debug';

        // no severity found
        return null;
      } catch (e) {
        return null;
      }
    }

    function createLineElement(text) {
      const div = document.createElement('div');
      div.className = 'line';

      const severity = detectSeverity(text);
      const useColor = colorizeCb && colorizeCb.checked;

      if (severity) {
        // Add a severity class on the line so CSS can target both badge and
        // the text. This also makes it easier to unit-test detection via DOM
        // inspection.
        div.classList.add('sev-' + severity);
      }

      if (useColor && severity) {
        const badge = document.createElement('span');
        badge.className = 'badge-sev sev-' + severity;
        badge.setAttribute('role', 'status');
        badge.setAttribute('aria-label', severity);

        const icon = document.createElement('span');
        icon.className = 'sev-icon';
        // Use simple ASCII-like icons to keep broad compatibility
        icon.textContent = severity === 'error' ? '!' : (severity === 'warning' ? '!' : (severity === 'info' ? 'i' : 'd'));
        badge.appendChild(icon);

        const label = document.createElement('span');
        label.className = 'sev-label';
        label.textContent = severity.toUpperCase();
        badge.appendChild(label);

        div.appendChild(badge);
      }

      const textSpan = document.createElement('div');
      textSpan.className = 'line-text';
      textSpan.textContent = text;
      div.appendChild(textSpan);

      return div;
    }

    function appendLine(text) {
      try {
        const el = createLineElement(text);
        logEl.appendChild(el);

        // Keep the page-level colorization state in sync. We toggle this
        // on the root container so text colour rules apply to existing lines
        // when the user toggles the checkbox.
        if (colorizeCb && colorizeCb.checked) {
          document.documentElement.classList.add('colorize-on');
        } else {
          document.documentElement.classList.remove('colorize-on');
        }

        if (autoscrollCb.checked) {
          logEl.scrollTop = logEl.scrollHeight;
        }
      } catch (e) {
        // Fallback to simple append if something goes wrong
        const div = document.createElement('div');
        div.className = 'line';
        div.textContent = text;
        logEl.appendChild(div);
      }
    }

    // Append a line with an explicit severity hint (used when server provides severity)
    function appendLineWithSeverity(text, severity) {
      try {
        // If severity is provided, create element similarly to createLineElement but skip re-detection
        const div = document.createElement('div');
        div.className = 'line';
        if (severity) {
          div.classList.add('sev-' + severity);
        }

        const useColor = colorizeCb && colorizeCb.checked;
        if (useColor && severity) {
          const badge = document.createElement('span');
          badge.className = 'badge-sev sev-' + severity;
          badge.setAttribute('role', 'status');
          badge.setAttribute('aria-label', severity);

          const icon = document.createElement('span');
          icon.className = 'sev-icon';
          icon.textContent = severity === 'error' ? '!' : (severity === 'warning' ? '!' : (severity === 'info' ? 'i' : 'd'));
          badge.appendChild(icon);

          const label = document.createElement('span');
          label.className = 'sev-label';
          label.textContent = (severity || '').toUpperCase();
          badge.appendChild(label);

          div.appendChild(badge);
        }

        const textSpan = document.createElement('div');
        textSpan.className = 'line-text';
        textSpan.textContent = text;
        div.appendChild(textSpan);

        logEl.appendChild(div);

        if (colorizeCb && colorizeCb.checked) {
          document.documentElement.classList.add('colorize-on');
        } else {
          document.documentElement.classList.remove('colorize-on');
        }

        if (autoscrollCb.checked) {
          logEl.scrollTop = logEl.scrollHeight;
        }
      } catch (e) {
        appendLine(text);
      }
    }

    // Ordered endpoint definitions (label and path to match)
    const endpointDefs = [
      { label: 'Chat', path: '/v1/chat/completions' },
      { label: 'Completions', path: '/v1/completions' },
      { label: 'Embeddings', path: '/v1/embeddings' },
      { label: 'Models', path: '/v1/models' }
    ];

    // Latest snapshots
    let latestCounts = {};
    let latestTokens = {};

    // Render unified summary for counts and tokens in a consistent order
    function renderSummary() {
      try {
        // Counts pane
        const countsEl = document.getElementById('counts');
        // Tokens pane
        const tokensEl = document.getElementById('tokens');

        const counts = latestCounts || {};
        const tokens = latestTokens || {};

        const countsParts = [];
        const tokensParts = [];

        for (const def of endpointDefs) {
          const label = def.label;
          const path = def.path;

          // Sum request counts for this path (match any key containing path)
          let reqTotal = 0;
          for (const [k,v] of Object.entries(counts)) {
            try {
              // Extract the path portion from the stored key (format: "METHOD <path> -> ...")
              const m = k.match(/^[A-Z]+\s+(\S+)\s+->/);
              const reqPath = m ? m[1] : null;
              const pathNoV1 = path.replace(/^\/v1\//, '/');

              // Only count when the recorded key's path exactly matches this endpoint's path
              // (either full /v1/... or the path without the /v1 prefix). Also ignore the
              // endpoint+model keys (those with '-> model:') to avoid double-counting.
              if (reqPath && (reqPath === path || reqPath === pathNoV1) && !k.includes('-> model:')) {
                reqTotal += Number(v || 0);
              }
            } catch (e) { /* ignore */ }
          }

          // Sum tokens sent/recv for this path (token keys are like 'sent:...')
          let sent = 0, recv = 0;
          for (const [k,v] of Object.entries(tokens)) {
            try {
              if (!k) continue;
              const n = Number(v || 0);
              if (k.startsWith('sent:') && k.includes(path)) sent += n;
              if (k.startsWith('recv:') && k.includes(path)) recv += n;
            } catch (e) { /* ignore */ }
          }

          countsParts.push(`<div class="line">${label}: <strong>${reqTotal}</strong></div>`);
          tokensParts.push(`<div class="line">${label}: <strong>sent ${sent}</strong> <span style="margin-left:8px;">recv <strong>${recv}</strong></span></div>`);
        }


        if (countsEl) countsEl.innerHTML = countsParts.join('') || '<div class="muted">No requests recorded yet.</div>';
        if (tokensEl) tokensEl.innerHTML = tokensParts.join('') || '<div class="muted">No token stats yet.</div>';

        // Also update raw debug views if present
        const rawCountsEl = document.getElementById('rawCounts');
        const rawTokensEl = document.getElementById('rawTokens');
        if (rawCountsEl) rawCountsEl.textContent = JSON.stringify(counts, null, 2);
        if (rawTokensEl) rawTokensEl.textContent = JSON.stringify(tokens, null, 2);
      } catch (e) { /* ignore */ }
    }

    // Render initial stats if available (injected server-side)
    try {
      if (window.__INITIAL_STATS) {
        latestCounts = window.__INITIAL_STATS.counts || {};
        latestTokens = window.__INITIAL_STATS.tokens || {};
        renderSummary();
      }
    } catch (e) { /* ignore */ }

    function connect() {
      if (es) return;
      const n = Math.max(1, parseInt(linesInput.value || '200'));
      const url = '/logs/tail?lines=' + encodeURIComponent(n);
      es = new EventSource(url);
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;

      es.onmessage = e => {
        try {
          const obj = JSON.parse(e.data);
          if (obj.initial) {
            appendLine('--- initial log ---');
            // Initial may be either a string (legacy) or an array of {line,severity}
            if (typeof obj.initial === 'string') {
              obj.initial.split(String.fromCharCode(10)).forEach(l => appendLine(l));
            } else if (Array.isArray(obj.initial)) {
              obj.initial.forEach(item => {
                try {
                  if (item && typeof item === 'object' && 'line' in item) {
                    const sev = item.severity || null;
                    appendLineWithSeverity(item.line, sev);
                  } else if (typeof item === 'string') {
                    appendLine(item);
                  }
                } catch (e) { appendLine(item.line || JSON.stringify(item)); }
              });
            }
            appendLine('--- end initial ---');
          } else if (obj.line) {
            // obj.line may be string (legacy) or present with severity
            if (typeof obj.line === 'string') {
              // Try to use provided severity if present on the object
              if ('severity' in obj && obj.severity) {
                appendLineWithSeverity(obj.line, obj.severity);
              } else {
                appendLine(obj.line);
              }
            } else if (typeof obj.line === 'object' && obj.line !== null) {
              // New shape: {line: '...', severity: 'info'}
              appendLineWithSeverity(obj.line.line || String(obj.line), obj.line.severity || null);
            }
          } else if (obj.counts) {
            // Update counts snapshot and re-render unified summary
            try {
              latestCounts = obj.counts || {};
              renderSummary();
            } catch (e) { /* ignore */ }
          } else if (obj.tokens) {
            // Update tokens snapshot and re-render unified summary
            try {
              latestTokens = obj.tokens || {};
              renderSummary();
            } catch (e) { /* ignore */ }
          } else if (obj.info) {
            appendLine('[info] ' + obj.info);
          } else if (obj.error) {
            appendLine('[error] ' + JSON.stringify(obj));
          }
        } catch (err) {
          // If parsing fails, append raw data
          try { appendLine(e.data); } catch (e) { /* ignore */ }
        }
      };

      es.onerror = () => {
        appendLine('[connection closed]');
        disconnect();
      };
    }

    function disconnect() {
      if (!es) return;
      es.close();
      es = null;
      connectBtn.disabled = false;
      disconnectBtn.disabled = true;
    }

    connectBtn.addEventListener('click', connect);
    disconnectBtn.addEventListener('click', disconnect);
    clearBtn.addEventListener('click', () => logEl.innerHTML = '');
    downloadBtn.addEventListener('click', () => {
      const text = Array.from(logEl.querySelectorAll('.line')).map(n => n.textContent).join(String.fromCharCode(10));
      const blob = new Blob([text], {type: 'text/plain'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'proxy.log'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    });

    // Auto-connect on load
    connect();
    window.addEventListener('beforeunload', disconnect);
  </script>
</body>
</html>"""

    # Prepare JSON snapshot for client-side rendering
    # Use shallow copies under locks
    async with counts_lock:
        counts_snapshot = dict(request_counts)
    async with token_lock:
        tokens_snapshot = dict(token_counts)

    model_list = list(config.get("models", {}).keys())
    model_list_json = json.dumps(model_list)
    initial_stats_json = json.dumps({"counts": counts_snapshot, "tokens": tokens_snapshot})

    # Replace placeholders with empty containers; client will render using INITIAL_STATS
    html = html.replace('{counts_html}', '')
    html = html.replace('{tokens_html}', '')

    # Inject initial stats and model list script before </body>
    html = html.replace('</body>', f'<script>window.__INITIAL_STATS = {initial_stats_json}; window.__MODEL_LIST = {model_list_json};</script></body>')
    return HTMLResponse(content=html)


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """
    Dedicated endpoint for embeddings requests.
    Validates the request and routes to the appropriate backend.
    
    The OpenAI embeddings API expects:
    - model: string (required)
    - input: string or array of strings (required)
    - encoding_format: string (optional, "float" or "base64")
    - dimensions: integer (optional)
    - user: string (optional)
    """
    # Parse request body
    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=400,
            detail="Request body is required"
        )
    
    try:
        body_json = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in request body"
        )
    
    # Validate required fields
    if "input" not in body_json:
        raise HTTPException(
            status_code=400,
            detail="'input' field is required for embeddings requests"
        )
    
    input_value = body_json["input"]
    if not isinstance(input_value, (str, list)):
        raise HTTPException(
            status_code=400,
            detail="'input' must be a string or an array of strings"
        )
    
    if isinstance(input_value, list):
        if len(input_value) == 0:
            raise HTTPException(
                status_code=400,
                detail="'input' array must not be empty"
            )
        if not all(isinstance(item, (str, int, list)) for item in input_value):
            raise HTTPException(
                status_code=400,
                detail="'input' array elements must be strings, integers, or arrays"
            )
    
    # Resolve model
    model_name = body_json.get("model")
    if not model_name and current_model:
        model_name = current_model
    
    model_cfg = get_model_config(model_name) if model_name else None
    
    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await proxy_to_remote(request, "v1/embeddings", default_remote)
        
        # If we have a current model loaded, try local
        if current_model:
            return await proxy_to_local(request, "v1/embeddings")
        
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )
    
    if model_cfg.get("type") == "local":
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        
        if current_model != llama_model:
            logger.info(
                f"Model switch requested for embeddings: {current_model} -> {llama_model}"
            )
            
            if not await ensure_model_loaded(model_name):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "model_loading",
                        "message": f"Failed to load model {model_name}",
                        "retry_after": 30
                    },
                    headers={"Retry-After": "30"}
                )
        
        return await proxy_to_local(request, "v1/embeddings")
    
    elif model_cfg.get("type") == "remote":
        return await proxy_to_remote(request, "v1/embeddings", model_cfg)
    
    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_openai_api(request: Request, path: str):
    """
    Main proxy endpoint for OpenAI API requests.
    Routes to local llama-server or remote API based on model.
    """
    # Get the request body to determine the model
    body = await request.body()
    body_json = {}
    model_name = None
    
    if body:
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
        except json.JSONDecodeError:
            pass
    
    # If no model specified, use the currently loaded model
    if not model_name and current_model:
        model_name = current_model
    
    # Get model configuration
    model_cfg = get_model_config(model_name) if model_name else None
    
    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await proxy_to_remote(request, f"v1/{path}", default_remote)
        
        # If we have a current model loaded, use that
        if current_model:
            return await proxy_to_local(request, f"v1/{path}")
        
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )
    
    if model_cfg.get("type") == "local":
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        
        if current_model != llama_model:
            logger.info(
                f"Model switch requested: {current_model} -> {llama_model}"
            )
            
            # Return 503 with Retry-After header
            # This tells the client to retry after the model loads
            if not await ensure_model_loaded(model_name):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "model_loading",
                        "message": f"Failed to load model {model_name}",
                        "retry_after": 30
                    },
                    headers={"Retry-After": "30"}
                )
        
        return await proxy_to_local(request, f"v1/{path}")
    
    elif model_cfg.get("type") == "remote":
        return await proxy_to_remote(request, f"v1/{path}", model_cfg)
    
    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )


@app.post("/admin/reload-config")
async def reload_config():
    """Reload configuration file."""
    global config
    try:
        config = load_config()
        logger.info("Configuration reloaded")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/switch-model/{model_name}")
async def switch_model(model_name: str):
    """Manually switch to a different model."""
    model_cfg = get_model_config(model_name)
    
    if model_cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    if model_cfg.get("type") != "local":
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} is not a local model"
        )
    
    if await ensure_model_loaded(model_name):
        return {
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "current_model": current_model
        }
    else:
        # Return the last captured start failure when available to aid UI debugging
        detail_msg = f"Failed to switch to model: {model_name}"
        if last_start_failure:
            detail_msg = detail_msg + "\n\nLast start failure:\n" + last_start_failure
        raise HTTPException(
            status_code=500,
            detail=detail_msg
        )


@app.post("/admin/stop-server")
async def admin_stop_server():
    """Stop the llama-server."""
    stop_llama_server()
    return {"status": "success", "message": "llama-server stopped"}


@app.get("/admin/dump-counts")
async def admin_dump_counts():
    """Return in-memory request and token counts for debugging."""
    # Snapshot under locks to avoid races
    snap_c = {}
    snap_t = {}
    async with counts_lock:
        snap_c = dict(request_counts)
    async with token_lock:
        snap_t = dict(token_counts)
    return {"counts": snap_c, "tokens": snap_t}


@app.post("/admin/reset-counts")
async def admin_reset_counts():
    """Reset in-memory and persisted request/token counts to empty.

    This clears the in-memory dictionaries and triggers an immediate persist.
    """
    global request_counts, token_counts, counts_dirty, tokens_dirty
    async with counts_lock:
        request_counts = {}
        counts_dirty = True
    async with token_lock:
        token_counts = {}
        tokens_dirty = True

    # Trigger async saves (background)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(save_counts())
        loop.create_task(save_token_counts())
    except RuntimeError:
        # fallback
        await save_counts()
        await save_token_counts()

    # Broadcast an empty snapshot to log tail clients
    for q in list(log_tail_clients):
        try:
            q.put_nowait({"counts": {}, "tokens": {}})
        except Exception:
            continue

    return {"status": "success", "message": "Counts reset"}


def main():
    """Main entry point."""
    import uvicorn
    
    # Load config for server settings
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
