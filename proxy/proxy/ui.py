"""
Web UI Module

Handles HTML endpoints and Web UI for the proxy server.
Uses lazy server import (_srv()) to avoid circular imports.
"""

import asyncio
import httpx
import json
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from proxy.lifecycle import _extract_router_model_ids
from proxy.observability import _build_llama_url, _query_slots_detail
from proxy.prompt_resolver import compose_messages, resolve_system_prompt
from proxy.provider import get_local_model_name_from_providers, get_model_type, get_remote_endpoint
from proxy.router_helpers import _get_per_model_queries


# ---------------------------------------------------------------------------
# Lazy server import
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


def _has_fallback_providers(model_cfg):
    """Check if a model config has providers that need fallback routing.

    Returns ``True`` when there are multiple providers or at least one
    remote provider — scenarios where ``proxy_with_fallback`` is needed
    to cascade through alternatives.  A single local provider does NOT
    need fallback; direct ``proxy_to_local`` preserves the original
    response (e.g., a transient 503) instead of converting it to the
    generic "All providers exhausted" error.
    """
    providers = model_cfg.get("providers") or []
    if not isinstance(providers, list):
        return False
    if len(providers) > 1:
        return True
    first = providers[0] if providers else {}
    return isinstance(first, dict) and first.get("type") == "remote"


def _build_home_model_rows(srv) -> str:
    """Build the Home tab model endpoint table rows.

    Returns an HTML string with one table row per provider entry in the
    fallback chain (not one row per model config key).  The table shows:
    - Model (1st col): the config key (rowspans across providers in chain)
    - Type: provider type (local/remote)
    - Primary Endpoint: the endpoint URL for this specific provider
    - Model (4th col): the provider-level model name (llama_model for
      local, endpoint model for remote)
    """
    rows = ""
    model_type_labels = {"local": "Local", "remote": "Remote"}
    model_type_badges = {"local": "badge-type-local", "remote": "badge-type-remote"}
    for name, cfg in srv.config.get("models", {}).items():
        providers = cfg.get("providers") or []
        provider_count = 0
        for p in providers:
            if isinstance(p, dict):
                provider_count += 1

        if provider_count == 0:
            continue

        # Type badge for this config key (first row only)
        model_type = get_model_type(cfg) or "unknown"
        type_label = model_type_labels.get(model_type, model_type.title())
        type_badge_class = model_type_badges.get(model_type, "badge-type-unknown")

        rowspan_attr = f' rowspan="{provider_count}"' if provider_count > 1 else ""

        for idx, p in enumerate(providers):
            if not isinstance(p, dict):
                continue

            ptype = p.get("type", "")
            endpoint = p.get("endpoint") or p.get("llama_model", "") or "-"
            provider_model_name = p.get("model") or p.get("llama_model") or p.get("name", "") or "-"
            is_first = idx == 0

            # Build the endpoint display
            endpoint_display = (
                f'<span class="provider-endpoint">{endpoint}</span>'
            )

            # Model name (4th col): primary at top, fallbacks below with indent
            if is_first:
                model_display = (
                    f'<span class="provider-model primary-model">{provider_model_name}</span>'
                )
            else:
                model_display = (
                    f'<span class="provider-model fallback-model">'
                    f'  <span class="fallback-indicator">↳ fallback:</span> {provider_model_name}'
                    f'</span>'
                )

            if is_first:
                rows += f"""
        <tr>
            <td{rowspan_attr}><code>{name}</code></td>
            <td{rowspan_attr}><span class="badge-type {type_badge_class}">{type_label}</span></td>
            <td>{endpoint_display}</td>
            <td>{model_display}</td>
        </tr>"""
            else:
                rows += f"""
        <tr>
            <td>{endpoint_display}</td>
            <td>{model_display}</td>
        </tr>"""
    return rows


async def index(request: Request):
    """Serve the index page with API documentation."""
    # Build models table rows and quick link buttons for local models
    srv = _srv()
    models_rows = ""
    model_buttons = ""
    model_options = ""
    for name, cfg in srv.config.get("models", {}).items():
        model_type = get_model_type(cfg) or "unknown"
        aliases = ", ".join(cfg.get("aliases", [])) or "—"
        endpoint = get_remote_endpoint(cfg) if model_type == "remote" else "Local llama-server"
        type_badge = '<span class="badge badge-local">Local</span>' if model_type == "local" else ('<span class="badge badge-remote">Remote</span>' if model_type == "remote" else '<span class="badge badge-unknown">Unknown</span>')

        # Build model dropdown options
        # Consider both the config key (name) and the underlying llama_model when
        # deciding which option is selected so UIs that compare against the
        # resolved llama-server id still show the correct active model.
        selected = ""
        try:
            lm = get_local_model_name_from_providers(cfg)
            if not lm:
                lm = name
            if name == srv.current_model or lm == srv.current_model:
                selected = "selected"
        except Exception:
            selected = "selected" if name == srv.current_model else ""
        type_label = "Local" if model_type == "local" else "Remote"
        model_options += f'<option value="{name}" {selected}>{name} ({type_label})</option>'

        # Add switch button for local models that aren't currently loaded
        action_cell = ""
        if model_type == "local":
            llama_model = get_local_model_name_from_providers(cfg)
            if not llama_model:
                llama_model = name
            # Consider model active when either the user-visible name or the
            # resolved llama_model matches the current_model state.
            if llama_model != srv.current_model and name != srv.current_model:
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
    local_model_names = [name for name, cfg in srv.config.get("models", {}).items() if get_model_type(cfg) == "local"]
    local_model_names_json = json.dumps(local_model_names)

    router_mode = srv.config.get("server", {}).get("llama_router_mode", False)
    router_models = None
    if router_mode:
        router_models = await srv.router_list_models()

    # Prefer configured provider host when present (e.g. Tailscale mapping)
    providers_cfg = srv.config.get('providers') if isinstance(srv.config.get('providers'), dict) else {}
    proxy_cfg = providers_cfg.get('Proxy') if providers_cfg else None
    provider_host = None
    if isinstance(proxy_cfg, dict):
        provider_host = proxy_cfg.get('host') or proxy_cfg.get('url') or proxy_cfg.get('base')
    provider_host_html = f'<div class="status-item"><strong>Provider:</strong> <code id="providerHost">{provider_host}</code></div>' if provider_host else ''
    # Base URL from incoming request (includes scheme and host:port)
    base = provider_host.rstrip('/') if provider_host else str(request.base_url).rstrip('/')

    # Load template from external file
    _templates_dir = Path(__file__).parent.parent / "templates"
    html_content = (_templates_dir / "index.html").read_text(encoding="utf-8")

    # Substitute placeholders
    html_content = html_content.replace('__PROVIDER_HOST_HTML__', provider_host_html)
    html_content = html_content.replace('__CURRENT_MODEL_DISPLAY__', srv.current_model or 'None')
    html_content = html_content.replace('__ROUTER_MODE_STR__', 'true' if router_mode else 'false')
    html_content = html_content.replace('__ROUTER_MODE_DISPLAY__', 'flex' if router_mode else 'none')
    html_content = html_content.replace('__ROUTER_MODE_LABEL__', 'Enabled' if router_mode else 'Disabled')
    html_content = html_content.replace('__LLAMA_STATUS_DISPLAY__', 'Running' if srv.llama_process and srv.llama_process.poll() is None else 'Stopped')
    html_content = html_content.replace('__MODEL_BUTTONS__', model_buttons)
    html_content = html_content.replace('__MODELS_ROWS__', models_rows)
    html_content = html_content.replace('__HOME_MODEL_ROWS__', _build_home_model_rows(srv))
    html_content = html_content.replace('__MODEL_OPTIONS__', model_options)
    html_content = html_content.replace('__CURRENT_MODEL_JS__', srv.current_model or 'None')
    html_content = html_content.replace('__LOCAL_MODEL_NAMES_JSON__', local_model_names_json)
    html_content = html_content.replace('__BASE__', base)
    html_content = html_content.replace('__LLAMA_SERVER_VERSION__', srv.llama_server_version)
    html_content = html_content.replace('__ROCM_VERSION__', srv.rocm_version)

    # Inject router script
    router_script = f'<script>window.__ROUTER_MODE = {json.dumps(router_mode)}; window.__ROUTER_MODELS = {json.dumps(router_models)};</script>'
    html_content = html_content.replace('__ROUTER_SCRIPT__', router_script)

    # (removed: per-provider model endpoint JSON – no longer needed)

    return HTMLResponse(content=html_content)









async def status_events():
    """Server-Sent Events endpoint for real-time status updates."""
    srv = _srv()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    srv.sse_clients.add(queue)

    async def event_generator():
        try:
            llama_status = await srv.query_llama_status()
            total_sent = srv.token_counts.get("total_sent", 0)
            total_recv = srv.token_counts.get("total_recv", 0)

            loaded_models = None
            if llama_status.get("router_mode"):
                router_models = await srv.router_list_models()
                loaded_models = _extract_router_model_ids(router_models)

            per_model_queries = await _get_per_model_queries(srv)

            # --- Per-slot data query (best-effort) ---
            slot_details = []
            if llama_status.get("llama_server_running"):
                try:
                    server_cfg = srv.config.get("server", {})
                    llama_port = int(server_cfg.get("llama_server_port", 8080) or 8080)
                    model_name = srv.current_model or None
                    slot_details = await _query_slots_detail(
                        llama_port, timeout=2.0, model=model_name,
                    )
                except Exception:
                    pass

            initial_status = json.dumps({
                "type": "status",
                "current_model": srv.current_model,
                "loaded_models": loaded_models,
                "llama_server_running": llama_status["llama_server_running"],
                "n_ctx": llama_status["n_ctx"],
                "kv_cache_tokens": llama_status["kv_cache_tokens"],
                "total_sent": total_sent,
                "total_recv": total_recv,
                "per_model_queries": per_model_queries,
                "slots": slot_details,
            })
            yield f"data: {initial_status}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield message
                except TimeoutError:
                    # keepalive comment
                    yield ": keepalive\n\n"
        finally:
            srv.sse_clients.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )






async def tail_logs(request: Request, lines: int = 100, source: str = "proxy"):
    """Stream a log file as Server-Sent Events (SSE).

    Query params:
    - lines: number of previous lines to include initially (default 100)
    - source: which log to tail: 'proxy' (default) or 'llama' for llama-server.log

    Sends an initial SSE message with key `initial` containing the last
    `lines` lines, then streams new lines as they are appended with key
    `line`. Includes a `source` field to identify which log the data belongs to.
    """
    # Validate source parameter
    srv = _srv()
    if source not in ("proxy", "llama"):
        source = "proxy"

    log_path = srv._resolve_log_path(source)

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

            # Send initial block of lines
            initial = await asyncio.to_thread(read_last_n, lines)
            yield f"data: {json.dumps({'initial': initial, 'source': source})}\n\n"

            # Register for counts updates
            counts_queue: asyncio.Queue | None = None
            try:
                counts_queue = asyncio.Queue(maxsize=10)
                srv.log_tail_clients.add(counts_queue)
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
                        except TimeoutError:
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
                    yield f"data: {json.dumps({'info': 'log_rotated_or_removed', 'source': source})}\n\n"
                    break

                cur_size = cur_stat.st_size
                if cur_size < last_pos:
                    # File truncated/rotated
                    last_pos = 0

                if cur_size > last_pos:
                    # Read new data
                    with open(log_path, encoding="utf-8", errors="replace") as f:
                        f.seek(last_pos)
                        new = f.read()
                    last_pos = cur_size

                    # Send each new line as its own SSE message
                    for line in new.splitlines():
                        yield f"data: {json.dumps({'line': line, 'source': source})}\n\n"
                else:
                    # No new file data; send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Cleanup
            try:
                if _local_counts_queue is not None:
                    srv.log_tail_clients.discard(_local_counts_queue)
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



async def view_logs(request: Request):
    """Simple web UI to view both proxy and llama-server logs using SSE from /logs/tail."""
    srv = _srv()
    _base = str(request.base_url).rstrip('/')
    async def get_counts_html():
        items = []
        async with srv.counts_lock:
            for k, v in srv.request_counts.items():
                items.append((k, v))
        items.sort(key=lambda x: (-x[1], x[0]))
        rows = '\n'.join([f'<div class="line">{k}: <strong>{v}</strong></div>' for k, v in items])
        if not rows:
            rows = '<div class="muted">No requests recorded yet.</div>'
        return rows

    _counts_html = await get_counts_html()
    async def get_tokens_html():
        items = []
        async with srv.token_lock:
            for k, v in srv.token_counts.items():
                items.append((k, v))
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

    _tokens_html = await get_tokens_html()

    # Load template from external file
    _templates_dir = Path(__file__).parent.parent / "templates"
    html = (_templates_dir / "view_logs.html").read_text(encoding="utf-8")

    # Prepare JSON snapshot for client-side rendering
    # Use shallow copies under locks
    async with srv.counts_lock:
        counts_snapshot = dict(srv.request_counts)
    async with srv.token_lock:
        tokens_snapshot = dict(srv.token_counts)

    initial_stats_json = json.dumps({"counts": counts_snapshot, "tokens": tokens_snapshot})

    # Replace placeholders with empty containers; client will render using INITIAL_STATS
    html = html.replace('{counts_html}', '')
    html = html.replace('{tokens_html}', '')

    # Inject version info placeholders
    html = html.replace('__LLAMA_SERVER_VERSION__', srv.llama_server_version)
    html = html.replace('__ROCM_VERSION__', srv.rocm_version)

    # Inject initial stats script before </body>
    log_viewer_script = f'<script>window.__INITIAL_STATS = {initial_stats_json};</script>'
    html = html.replace('__LOG_VIEWER_SCRIPT__', log_viewer_script)

    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Shared local-model dispatch orchestration
# ---------------------------------------------------------------------------


async def _dispatch_local_model_load(
    request: Request,
    srv,
    model_cfg: dict,
    model_name: str,
    endpoint_path: str,
    *,
    enable_grace_window: bool = False,
    grace_window_seconds: float = 0.75,
):
    """Shared local-model load orchestration for embeddings and chat handlers.

    Handles the decision flow common to both endpoints:
    1. Active model fast path
    2. Router loaded-check fast path
    3. Background load scheduling
    4. Grace window (optional, used by chat but not embeddings)
    5. Session/slot restore detection (scheduler reenter + slot file)
    6. Remote provider fallback
    7. Final model_loading response

    Args:
        request: The incoming request.
        srv: The server module (from _srv()).
        model_cfg: The model configuration dict.
        model_name: The resolved model name from the request body.
        endpoint_path: The endpoint path (e.g. "v1/embeddings", "v1/chat/completions").
        enable_grace_window: Whether to enable the router-mode grace window.
        grace_window_seconds: Maximum seconds to wait for model in grace window.

    Returns:
        A Response (either direct proxy response or model_loading 503).
    """
    server_config = srv.config.get("server", {})
    router_mode = server_config.get("llama_router_mode", False)

    llama_model = get_local_model_name_from_providers(model_cfg)
    if not isinstance(llama_model, str) or not llama_model:
        raise HTTPException(
            status_code=500,
            detail=f"Local model configuration missing llama_model for: {model_name}",
        )
    llama_model_str: str = llama_model

    # If model already active and process running, proceed immediately
    if srv.current_model == llama_model_str and srv.llama_process is not None and (srv.llama_process.poll() is None):
        if _has_fallback_providers(model_cfg):
            from proxy.provider import proxy_with_fallback
            return await proxy_with_fallback(request, endpoint_path, model_cfg, srv.config)
        return await srv.proxy_to_local(request, endpoint_path)

    # Try a fast router-mode check: model may already be loaded in router
    if router_mode:
        try:
            if await srv.router_is_model_loaded(llama_model_str):
                srv.logger.info(f"Router reports model {llama_model_str} already loaded; serving request immediately")
                srv.current_model = llama_model_str
                if _has_fallback_providers(model_cfg):
                    from proxy.provider import proxy_with_fallback
                    return await proxy_with_fallback(request, endpoint_path, model_cfg, srv.config)
                return await srv.proxy_to_local(request, endpoint_path)
        except Exception:
            srv.logger.debug("Fast router check failed; scheduling background load")

    # Schedule background load
    target_model: str = model_name if isinstance(model_name, str) and model_name else llama_model_str
    scheduled = srv.schedule_background_load(target_model)
    srv.logger.info(f"Scheduled background load for request: model={target_model} scheduled={scheduled}")

    # In router mode, allow a short grace window for transient model-state lag
    if enable_grace_window and router_mode:
        grace_seconds = float(server_config.get("model_loading_local_grace_seconds", grace_window_seconds) or grace_window_seconds)
        grace_seconds = max(0.0, grace_seconds)
        if grace_seconds > 0:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + grace_seconds
            while loop.time() < deadline:
                try:
                    model_ready = await srv.router_is_model_loaded(llama_model_str)
                    if not model_ready:
                        try:
                            model_ready = await srv.router_load_model(llama_model_str)
                        except Exception:
                            model_ready = False

                    if model_ready:
                        srv.logger.info(
                            f"Router model {llama_model_str} became available during grace window; serving locally",
                        )
                        srv.current_model = llama_model_str
                        if _has_fallback_providers(model_cfg):
                            from proxy.provider import proxy_with_fallback
                            return await proxy_with_fallback(request, endpoint_path, model_cfg, srv.config)
                        return await srv.proxy_to_local(request, endpoint_path)
                except Exception:
                    srv.logger.debug("Grace-window router check failed; retrying", exc_info=True)
                await asyncio.sleep(0.1)

    # Session restore detection
    try:
        from proxy.session import _build_slot_context, _resolve_session_id_header
        session_id_header, _ = _resolve_session_id_header(request.headers)
        slot_id, slot_filename, _ = _build_slot_context(srv.config.get("server", {}), session_id_header)
        if slot_filename and slot_filename != "" and Path(slot_filename).exists():
            return srv._model_loading_response(
                requested_model=model_name if isinstance(model_name, str) else None,
                target_model=target_model,
                scheduled=scheduled,
                endpoint=f"/{endpoint_path}",
            )
    except Exception:
        srv.logger.debug("Session/slot detection failed; will attempt remote fallback before returning model_loading", exc_info=True)

    # Try configured remote providers first
    try:
        providers = model_cfg.get("providers") or []
        remote_providers = [p for p in providers if isinstance(p, dict) and p.get("type") == "remote"]
        if remote_providers:
            remote_cfg = {"providers": remote_providers}
            from proxy.provider import proxy_with_remote_fallback
            try:
                resp = await proxy_with_remote_fallback(request, endpoint_path, remote_cfg, srv.config)
                if resp.status_code >= 400:
                    srv.logger.warning(
                        f"Remote fallback returned error for model={model_name} status={resp.status_code}; "
                        "returning model_loading response",
                    )
                    return srv._model_loading_response(
                        requested_model=model_name if isinstance(model_name, str) else None,
                        target_model=target_model,
                        scheduled=scheduled,
                        endpoint=f"/{endpoint_path}",
                    )
                return resp
            except Exception:
                srv.logger.exception("Remote fallback attempt raised exception; returning model_loading response")
                return srv._model_loading_response(
                    requested_model=model_name if isinstance(model_name, str) else None,
                    target_model=target_model,
                    scheduled=scheduled,
                    endpoint=f"/{endpoint_path}",
                )
    except Exception:
        srv.logger.exception("Failed while attempting remote fallback; returning model_loading response")

    return srv._model_loading_response(
        requested_model=model_name if isinstance(model_name, str) else None,
        target_model=target_model,
        scheduled=scheduled,
        endpoint=f"/{endpoint_path}",
    )


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
    srv = _srv()
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
    if not model_name and srv.current_model:
        model_name = srv.current_model

    model_cfg = srv.get_model_config(model_name) if model_name else None

    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = srv.config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await srv.proxy_to_remote(request, "v1/embeddings", default_remote)

        # If we have a current model loaded, try local
        if srv.current_model:
            return await srv.proxy_to_local(request, "v1/embeddings")

        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )

    if get_model_type(model_cfg) == "local":
        return await _dispatch_local_model_load(request, srv, model_cfg, model_name, "v1/embeddings")

    elif get_model_type(model_cfg) == "remote":
        providers = model_cfg.get("providers")
        if providers:
            from proxy.provider import proxy_with_remote_fallback
            return await proxy_with_remote_fallback(request, "v1/embeddings", model_cfg, srv.config)
        return await srv.proxy_to_remote(request, "v1/embeddings", model_cfg)

    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )



async def proxy_openai_api(request: Request, path: str):
    """
    Main proxy endpoint for OpenAI API requests.
    Routes to local llama-server or remote API based on model.

    Duplicate in-flight requests are coalesced via ``RequestCoalescer``
    to prevent retry cascades when the Pi client retries a request that
    is still being processed.
    """
    from proxy.request_coalescer import get_coalescer

    # Read the body early so the coalescer has the hash key.
    srv = _srv()
    body = await request.body()

    # Wrap the rest of the processing in an inner coroutine so we can
    # pass it to the coalescer for deduplication.
    async def _process_request():
        return await _do_proxy_openai_api(request, path, body, srv)

    return await get_coalescer().coalesce_or_execute(path, body, _process_request)


async def _do_proxy_openai_api(
    request: Request,
    path: str,
    body: bytes,
    srv,
):
    """Inner implementation of proxy_openai_api.

    Extracted so that request coalescing can wrap just the outer call
    without interfering with the processing logic.
    """
    # Get the request body to determine the model
    body_json = {}
    model_name = None

    if body:
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
        except json.JSONDecodeError:
            pass

    # If no model specified, use the currently loaded model
    if not model_name and srv.current_model:
        model_name = srv.current_model

    # Get model configuration
    model_cfg = srv.get_model_config(model_name) if model_name else None

    # Apply system prompt if configured
    if model_cfg is not None and "messages" in body_json and isinstance(body_json.get("messages"), list):
        prompt_result = resolve_system_prompt(model_name, model_cfg) if model_name else None
        if prompt_result is not None and body_json["messages"]:
            original_count = len(body_json["messages"])
            body_json["messages"] = compose_messages(body_json["messages"], prompt_result)
            new_body = json.dumps(body_json).encode("utf-8")
            request._body = new_body
            body = new_body
            srv.logger.info(
                "Applied system_prompt mode=%s to %s messages (was %d, now %d)",
                prompt_result["mode"], model_name, original_count, len(body_json["messages"]),
            )

    if model_cfg is None:
        # Check if default remote is enabled
        default_remote = srv.config.get("default_remote", {})
        if default_remote.get("enabled", False):
            return await srv.proxy_to_remote(request, f"v1/{path}", default_remote)

        # If we have a current model loaded, use that
        if srv.current_model:
            return await srv.proxy_to_local(request, f"v1/{path}")

        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. No default remote configured."
        )

    if get_model_type(model_cfg) == "local":
        return await _dispatch_local_model_load(
            request, srv, model_cfg, model_name, f"v1/{path}",
            enable_grace_window=True,
        )

    elif get_model_type(model_cfg) == "remote":
        providers = model_cfg.get("providers")
        if providers:
            from proxy.provider import proxy_with_remote_fallback
            return await proxy_with_remote_fallback(request, f"v1/{path}", model_cfg, srv.config)
        return await srv.proxy_to_remote(request, f"v1/{path}", model_cfg)

    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )






async def switch_model(model_name: str):
    """Manually switch to a different model."""
    srv = _srv()
    model_cfg = srv.get_model_config(model_name)

    if model_cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    if get_model_type(model_cfg) != "local":
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not a local model"
        )

    srv.logger.info(f"Admin switch-model requested: {model_name}; current_model before: {srv.current_model}; llama_running: {srv.llama_process is not None and srv.llama_process.poll() is None}")
    if await srv.ensure_model_loaded(model_name):
        srv.logger.info(f"Admin switch-model succeeded: requested={model_name} current_model after: {srv.current_model}")
        return {
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "current_model": srv.current_model,
            "llama_server_running": srv.llama_process is not None and srv.llama_process.poll() is None,
            "last_start_failure": srv.last_start_failure,
        }
    else:
        # Return the last captured start failure when available to aid UI debugging
        detail_msg = f"Failed to switch to model: {model_name}"
        if srv.last_start_failure:
            detail_msg = detail_msg + "\n\nLast start failure:\n" + srv.last_start_failure
        srv.logger.error(f"Admin switch-model failed: {model_name}; reason: {detail_msg}")
        raise HTTPException(
            status_code=500,
            detail=detail_msg
        )


# ═══════════════════════════════════════════════════════════════════════════
# Admin endpoint for session recordings
# ═══════════════════════════════════════════════════════════════════════════

def _get_recorder():
    """Resolve the global SessionRecorder instance from the server module."""
    srv = _srv()
    if not hasattr(srv, "session_recorder") or srv.session_recorder is None:
        from proxy.session_recorder import SessionRecorder
        srv.session_recorder = SessionRecorder.from_config(srv.config)
    return srv.session_recorder


async def list_session_recordings(session_id: str) -> JSONResponse:
    """Return a list of recording files for a given session.

    Args:
        session_id: The session identifier to look up recordings for.

    Returns:
        JSONResponse with status 200 and a JSON array of recording metadata,
        or status 404 with a descriptive message if the session has no
        recordings.
    """
    recorder = _get_recorder()
    recordings = recorder.get_recordings_list(session_id)
    if not recordings:
        return JSONResponse(
            status_code=404,
            content={
                "error": "No recordings found for this session",
                "session_id": session_id,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "session_id": session_id,
            "recordings": recordings,
        },
    )


async def get_session_recording(session_id: str, filename: str) -> JSONResponse:
    """Return the content of a specific recording file.

    Args:
        session_id: The session identifier.
        filename: The base filename of the recording (e.g.,
            ``"2026-07-06T10:00:00.000000-request.json"``).

    Returns:
        JSONResponse with status 200 and the recording content,
        or status 404 if the file is not found.
    """
    recorder = _get_recorder()
    content = recorder.get_recording(session_id, filename)
    if content is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Recording not found",
                "session_id": session_id,
                "filename": filename,
            },
        )
    return JSONResponse(
        status_code=200,
        content=content,
    )


async def list_all_sessions(request: Request = None) -> JSONResponse:
    """Return all session IDs that have recordings, optionally filtered by model.

    Query params:
        model: Optional model name to filter sessions by.

    Returns recording-based sessions merged with live session-manager sessions.
    """
    model_filter = None
    if request is not None:
        model_filter = request.query_params.get("model")

    # Helper: merge recording sessions with live session-manager sessions
    async def _merge_live_and_recording_sessions(rec_sessions):
        live_sessions = []
        try:
            srv = _srv()
            if hasattr(srv, "session_manager") and srv.session_manager is not None:
                live_sessions = await srv.session_manager.list_sessions()
        except Exception:
            pass

        seen = set()
        merged = []
        # Live sessions first (mark as active)
        for s in live_sessions:
            sid = s.get("session_id", "")
            if sid:
                seen.add(sid)
                s["active"] = True
                merged.append(s)
        # Recording-only sessions (not already in live list, mark as inactive)
        for s in rec_sessions:
            sid = s.get("session_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                merged.append({
                    "session_id": sid,
                    "response_time": s.get("response_time", ""),
                    "last_activity": s.get("last_activity", s.get("response_time", "")),
                    "model": s.get("model", ""),
                    "provider": s.get("provider", ""),
                    "active": False,
                })
        return merged

    recorder = _get_recorder()
    rec_sessions = recorder.list_sessions_by_model(model_filter) if model_filter else recorder.list_sessions()
    merged = await _merge_live_and_recording_sessions(rec_sessions)

    # Also include recording sessions not already merged from the live list
    seen_live = {s.get("session_id", "") for s in merged}
    for s in rec_sessions:
        sid = s.get("session_id", "")
        if sid and sid not in seen_live:
            seen_live.add(sid)
            s["active"] = s.get("active", False)
            merged.append(s)

    # Sort all sessions by last_activity descending — the most recently updated
    # session appears at the top regardless of active/inactive status.
    merged.sort(key=lambda s: s.get("last_activity", s.get("response_time", "")), reverse=True)

    result = {"sessions": merged, "count": len(merged)}
    if model_filter:
        result["model"] = model_filter

    return JSONResponse(status_code=200, content=result)


def list_session_recording_routes(app):
    """Register session recording admin routes on a FastAPI application.

    Args:
        app: A FastAPI application instance.
    """
    app.add_api_route(
        "/admin/sessions",
        list_all_sessions,
        methods=["GET"],
        summary="List all sessions with recordings (filter by ?model=<name>)",
    )
    app.add_api_route(
        "/admin/sessions/{session_id}/recordings",
        list_session_recordings,
        methods=["GET"],
        summary="List recordings for a session",
    )
    app.add_api_route(
        "/admin/sessions/{session_id}/recordings/{filename:path}",
        get_session_recording,
        methods=["GET"],
        summary="Get a specific recording file",
    )

