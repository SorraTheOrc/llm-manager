"""
Web UI Module

Handles HTML endpoints and Web UI for the proxy server.
Uses lazy server import (_srv()) to avoid circular imports.
"""

import asyncio
import json
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Lazy server import
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


async def index(request: Request):
    """Serve the index page with API documentation."""
    # Build models table rows and quick link buttons for local models
    srv = _srv()
    models_rows = ""
    model_buttons = ""
    model_options = ""
    for name, cfg in srv.config.get("models", {}).items():
        model_type = cfg.get("type", "unknown")
        aliases = ", ".join(cfg.get("aliases", [])) or "—"
        endpoint = cfg.get("endpoint", "localhost:8080") if model_type == "remote" else "Local llama-server"
        type_badge = f'<span class="badge badge-local">Local</span>' if model_type == "local" else f'<span class="badge badge-remote">Remote</span>'
        
        # Build model dropdown options
        # Consider both the config key (name) and the underlying llama_model when
        # deciding which option is selected so UIs that compare against the
        # resolved llama-server id still show the correct active model.
        selected = ""
        try:
            lm = cfg.get("llama_model", name)
            if name == srv.current_model or lm == srv.current_model:
                selected = "selected"
        except Exception:
            selected = "selected" if name == srv.current_model else ""
        type_label = "Local" if model_type == "local" else "Remote"
        model_options += f'<option value="{name}" {selected}>{name} ({type_label})</option>'
        
        # Add switch button for local models that aren't currently loaded
        action_cell = ""
        if model_type == "local":
            llama_model = cfg.get("llama_model", name)
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
    local_model_names = [name for name, cfg in srv.config.get("models", {}).items() if cfg.get("type") == "local"]
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
            --log-error: #b00020;
            --log-warning: #f59e0b;
            --log-info: #60a5fa;
            --log-debug: #9ca3af;
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
            flex-wrap: wrap;
            gap: 1.5rem;
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
            {provider_host_html}
            <div class="status-item">
                <strong>Current Model:</strong>
                <code id="currentModelStatus">{srv.current_model or 'None'}</code>
            </div>
            <div class="status-item" id="routerModeStatus" data-router-mode="{'true' if router_mode else 'false'}" style="display: {'flex' if router_mode else 'none'};">
                <strong>Router:</strong>
                <span id="routerModeLabel">{'Enabled' if router_mode else 'Disabled'}</span>
            </div>
            <div class="status-item">
                <strong>llama-server:</strong>
                <span id="llamaServerStatus">{'Running' if srv.llama_process and srv.llama_process.poll() is None else 'Stopped'}</span>
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
                        <span class="endpoint-path">/admin/reload-srv.config</span>
                        <button class="btn-switch" style="margin-left:auto;" onclick="reloadConfig()">Reload</button>
                    </div>
                    <p class="endpoint-desc">Reload configuration from srv.config.yaml</p>
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
    Reload srv.config:  POST {base}/admin/reload-srv.config
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
                    statusEl.className = 'status-message loading';
                    statusEl.textContent = `Switch requested for ${{modelName}}. Waiting for readiness...`;
                    statusEl.style.display = 'block';
                    btn.disabled = false;
                    btn.textContent = 'Load Model';
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
        const routerModeEl = document.getElementById('routerModeStatus');
        const routerModeLabel = document.getElementById('routerModeLabel');
        
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
        const routerModeEnabled = Boolean(window.__ROUTER_MODE);
        const routerModels = window.__ROUTER_MODELS;

        if (routerModeEl) {{
            const serverFlag = routerModeEl.dataset.routerMode === 'true';
            const enabled = routerModeEnabled || serverFlag;
            routerModeEl.style.display = enabled ? 'flex' : 'none';
            if (routerModeLabel) routerModeLabel.textContent = enabled ? 'Enabled' : 'Disabled';
        }}
        
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

            if (data.loaded_models !== undefined) {{
                const val = data.loaded_models.length ? data.loaded_models.join(', ') : '-';
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
                if (data.loaded_models) {{
                    statsModelEl.textContent = data.loaded_models.join(', ');
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
                            if (data.loaded_models) {{
                                statsModelEl.textContent = data.loaded_models.join(', ');
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
                    statusEl.className = 'status-message loading';
                    statusEl.textContent = data.message || `Switch requested for ${{selected}}. Waiting for readiness...`;
                    statusEl.style.display = 'block';
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
    html_content = html_content.replace(
        '</body>',
        f'<script>window.__ROUTER_MODE = {json.dumps(router_mode)}; window.__ROUTER_MODELS = {json.dumps(router_models)};</script></body>'
    )
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

            initial_status = json.dumps({
                "type": "status",
                "current_model": srv.current_model,
                "loaded_models": loaded_models,
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
                    yield f"data: {json.dumps({'info': 'log_rotated_or_removed', 'source': source})}\n\n"
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
    base = str(request.base_url).rstrip('/')
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

    counts_html = await get_counts_html()
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

    tokens_html = await get_tokens_html()

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Viewer</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #4f8cff;
            --accent-hover: #6ba1ff;
            --accent-llama: #ff8c4f;
            --accent-llama-hover: #ffa66b;
            --success: #4caf50;
            --warning: #ff9800;
            --border: #2a3a5a;
            --log-error: #b00020;
            --log-warning: #f59e0b;
            --log-info: #60a5fa;
            --log-debug: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 1rem;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { display:flex; gap:1rem; align-items:center; margin-bottom:1rem; flex-wrap:wrap; }
        .header-left { display:flex; align-items:center; gap:1rem; }
        .controls { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }
        .controls-llama { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }
        input[type=number] { width:5rem; padding:0.35rem; border-radius:6px; border:1px solid var(--border); background:var(--bg-card); color:var(--text-primary); }
        button { border:none; padding:0.4rem 0.6rem; border-radius:6px; cursor:pointer; }
        button.proxy-btn { background:var(--accent); color:#fff; }
        button.proxy-btn:hover { background:var(--accent-hover); }
        button.llama-btn { background:var(--accent-llama); color:#fff; }
        button.llama-btn:hover { background:var(--accent-llama-hover); }
        button:disabled { opacity:0.5; cursor:not-allowed; }
        .log-panes { display:grid; grid-template-columns:1fr 1fr; gap:1rem; height: calc(100vh - 200px); }
        .pane { display:flex; flex-direction:column; overflow:hidden; }
        .pane-header { display:flex; align-items:center; gap:0.5rem; padding:0.5rem; background:var(--bg-card); border-radius:8px 8px 0 0; border:1px solid var(--border); border-bottom:none; }
        .pane-header.proxy { color:var(--accent); font-weight:600; }
        .pane-header.llama { color:var(--accent-llama); font-weight:600; }
        .pane-controls { display:flex; gap:0.3rem; align-items:center; flex-wrap:wrap; }
        .pane-controls input[type=number] { width:4rem; }
        .pane-controls button { padding:0.25rem 0.5rem; font-size:0.85rem; }
        .pane-controls button.proxy-btn { background:var(--accent); color:#fff; }
        .pane-controls button.llama-btn { background:var(--accent-llama); color:#fff; }
        .pane-controls button.connected.proxy-btn { background:var(--success); }
        .pane-controls button.connected.llama-btn { background:var(--success); }
        .log { flex:1; overflow:auto; padding:0.75rem; font-family: monospace; background: linear-gradient(180deg, var(--bg-card), rgba(15,18,30,1)); border:1px solid var(--border); border-radius:0 0 8px 8px; white-space:pre-wrap; }
        .line { padding:0 0 2px 0; border-bottom:1px solid rgba(255,255,255,0.02); font-size:0.85rem; }
        .muted { color:var(--text-secondary); font-size:0.9rem; }
        .summary { margin-bottom:0.75rem; }
        .summary h3 { margin:0 0 0.5rem 0; color:var(--accent); }
        .summary-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
        .summary-card { background:var(--bg-card); padding:0.75rem; border-radius:6px; border:1px solid var(--border); max-height:140px; overflow:auto; }
        .summary-card h4 { color:var(--text-secondary); font-size:0.85rem; margin-bottom:0.25rem; }
        .pane-label { font-size:0.9rem; min-width:80px; }
    </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <a href="/" style="color:var(--accent); text-decoration:none; font-weight:600;">Home</a>
      </div>
      <div style="margin-left:auto; display:flex; gap:0.5rem; align-items:center;">
        <label class="muted" style="font-size:0.85rem;">Shared Lines:</label>
        <input id="sharedLines" type="number" value="200" min="1" style="width:5rem;" />
      </div>
    </div>

    <div class="summary">
      <h3>Request Summary</h3>
      <div class="summary-grid">
        <div class="summary-card">
          <h4>Counts</h4>
          <div id="counts"></div>
          <pre id="rawCounts" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
        <div class="summary-card">
          <h4>Tokens</h4>
          <div id="tokens"></div>
          <pre id="rawTokens" style="display:none; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);"></pre>
        </div>
      </div>
    </div>

    <div class="log-panes">
      <!-- Proxy Log Pane -->
      <div class="pane" id="proxyPane">
        <div class="pane-header proxy">
          <span class="pane-label">Proxy log</span>
          <div class="pane-controls">
            <input id="proxyLines" type="number" value="200" min="1" />
            <button id="proxyConnect" class="proxy-btn">Connect</button>
            <button id="proxyDisconnect" class="proxy-btn" disabled>Disconnect</button>
            <button id="proxyClear" class="proxy-btn">Clear</button>
            <label style="display:flex; align-items:center; gap:0.25rem; color:var(--text-secondary); font-size:0.8rem;">
              <input id="proxyAutoscroll" type="checkbox" checked /> Auto
            </label>
            <button id="proxyDownload" class="proxy-btn">Download</button>
          </div>
        </div>
        <div id="proxyLog" class="log"></div>
      </div>

      <!-- Llama-server Log Pane -->
      <div class="pane" id="llamaPane">
        <div class="pane-header llama">
          <span class="pane-label">Llama-server log</span>
          <div class="pane-controls">
            <input id="llamaLines" type="number" value="200" min="1" />
            <button id="llamaConnect" class="llama-btn">Connect</button>
            <button id="llamaDisconnect" class="llama-btn" disabled>Disconnect</button>
            <button id="llamaClear" class="llama-btn">Clear</button>
            <label style="display:flex; align-items:center; gap:0.25rem; color:var(--text-secondary); font-size:0.8rem;">
              <input id="llamaAutoscroll" type="checkbox" checked /> Auto
            </label>
            <button id="llamaDownload" class="llama-btn">Download</button>
          </div>
        </div>
        <div id="llamaLog" class="log"></div>
      </div>
    </div>
  </div>

  <script>
    const endpointDefs = [
      { label: 'Chat', path: '/v1/chat/completions' },
      { label: 'Completions', path: '/v1/completions' },
      { label: 'Embeddings', path: '/v1/embeddings' },
      { label: 'Models', path: '/v1/models' }
    ];

    let latestCounts = {};
    let latestTokens = {};
    let esProxy = null;
    let esLlama = null;

    const proxyLog = document.getElementById('proxyLog');
    const llamaLog = document.getElementById('llamaLog');
    const proxyLinesInput = document.getElementById('proxyLines');
    const llamaLinesInput = document.getElementById('llamaLines');
    const sharedLinesInput = document.getElementById('sharedLines');
    const proxyConnectBtn = document.getElementById('proxyConnect');
    const proxyDisconnectBtn = document.getElementById('proxyDisconnect');
    const llamaConnectBtn = document.getElementById('llamaConnect');
    const llamaDisconnectBtn = document.getElementById('llamaDisconnect');
    const proxyAutoscrollCb = document.getElementById('proxyAutoscroll');
    const llamaAutoscrollCb = document.getElementById('llamaAutoscroll');

    // Sync shared lines input with individual inputs
    sharedLinesInput.addEventListener('change', () => {
      const n = sharedLinesInput.value;
      proxyLinesInput.value = n;
      llamaLinesInput.value = n;
    });
    proxyLinesInput.addEventListener('change', () => {
      sharedLinesInput.value = proxyLinesInput.value;
    });
    llamaLinesInput.addEventListener('change', () => {
      sharedLinesInput.value = llamaLinesInput.value;
    });

    function appendLine(logEl, autoscrollCb, text) {
      const div = document.createElement('div');
      div.className = 'line';
      div.textContent = text;
      logEl.appendChild(div);
      if (autoscrollCb.checked) {
        logEl.scrollTop = logEl.scrollHeight;
      }
    }

    function renderSummary() {
      try {
        const countsEl = document.getElementById('counts');
        const tokensEl = document.getElementById('tokens');
        const counts = latestCounts || {};
        const tokens = latestTokens || {};
        const countsParts = [];
        const tokensParts = [];

        for (const def of endpointDefs) {
          const label = def.label;
          const path = def.path;
          let reqTotal = 0;
          for (const [k,v] of Object.entries(counts)) {
            try {
              const m = k.match(/^[A-Z]+\s+(\S+)\s+->/);
              const reqPath = m ? m[1] : null;
              const pathNoV1 = path.replace(/^\/v1\//, '/');
              if (reqPath && (reqPath === path || reqPath === pathNoV1) && !k.includes('-> model:')) {
                reqTotal += Number(v || 0);
              }
            } catch (e) { /* ignore */ }
          }

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

        const rawCountsEl = document.getElementById('rawCounts');
        const rawTokensEl = document.getElementById('rawTokens');
        if (rawCountsEl) rawCountsEl.textContent = JSON.stringify(counts, null, 2);
        if (rawTokensEl) rawTokensEl.textContent = JSON.stringify(tokens, null, 2);
      } catch (e) { /* ignore */ }
    }

    try {
      if (window.__INITIAL_STATS) {
        latestCounts = window.__INITIAL_STATS.counts || {};
        latestTokens = window.__INITIAL_STATS.tokens || {};
        renderSummary();
      }
    } catch (e) { /* ignore */ }

    function handleMessage(logEl, autoscrollCb, obj) {
      if (obj.initial) {
        appendLine(logEl, autoscrollCb, '--- initial log ---');
        obj.initial.split(String.fromCharCode(10)).forEach(l => appendLine(logEl, autoscrollCb, l));
        appendLine(logEl, autoscrollCb, '--- end initial ---');
      } else if (obj.line) {
        appendLine(logEl, autoscrollCb, obj.line);
      } else if (obj.counts) {
        try {
          latestCounts = obj.counts || {};
          renderSummary();
        } catch (e) { /* ignore */ }
      } else if (obj.tokens) {
        try {
          latestTokens = obj.tokens || {};
          renderSummary();
        } catch (e) { /* ignore */ }
      } else if (obj.info) {
        appendLine(logEl, autoscrollCb, '[info] ' + obj.info);
      } else if (obj.error) {
        appendLine(logEl, autoscrollCb, '[error] ' + JSON.stringify(obj));
      }
    }

    function connectProxy() {
      if (esProxy) return;
      const n = Math.max(1, parseInt(proxyLinesInput.value || '200'));
      const url = '/logs/tail?lines=' + encodeURIComponent(n) + '&source=proxy';
      esProxy = new EventSource(url);
      proxyConnectBtn.disabled = true;
      proxyDisconnectBtn.disabled = false;
      proxyConnectBtn.classList.add('connected');

      esProxy.onmessage = e => {
        try {
          const obj = JSON.parse(e.data);
          handleMessage(proxyLog, proxyAutoscrollCb, obj);
        } catch (err) {
          appendLine(proxyLog, proxyAutoscrollCb, e.data);
        }
      };

      esProxy.onerror = () => {
        appendLine(proxyLog, proxyAutoscrollCb, '[connection closed]');
        disconnectProxy();
      };
    }

    function disconnectProxy() {
      if (!esProxy) return;
      esProxy.close();
      esProxy = null;
      proxyConnectBtn.disabled = false;
      proxyDisconnectBtn.disabled = true;
      proxyConnectBtn.classList.remove('connected');
    }

    function connectLlama() {
      if (esLlama) return;
      const n = Math.max(1, parseInt(llamaLinesInput.value || '200'));
      const url = '/logs/tail?lines=' + encodeURIComponent(n) + '&source=llama';
      esLlama = new EventSource(url);
      llamaConnectBtn.disabled = true;
      llamaDisconnectBtn.disabled = false;
      llamaConnectBtn.classList.add('connected');

      esLlama.onmessage = e => {
        try {
          const obj = JSON.parse(e.data);
          handleMessage(llamaLog, llamaAutoscrollCb, obj);
        } catch (err) {
          appendLine(llamaLog, llamaAutoscrollCb, e.data);
        }
      };

      esLlama.onerror = () => {
        appendLine(llamaLog, llamaAutoscrollCb, '[connection closed]');
        disconnectLlama();
      };
    }

    function disconnectLlama() {
      if (!esLlama) return;
      esLlama.close();
      esLlama = null;
      llamaConnectBtn.disabled = false;
      llamaDisconnectBtn.disabled = true;
      llamaConnectBtn.classList.remove('connected');
    }

    function downloadLog(logEl, filename) {
      const text = Array.from(logEl.querySelectorAll('.line')).map(n => n.textContent).join(String.fromCharCode(10));
      const blob = new Blob([text], {type: 'text/plain'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    proxyConnectBtn.addEventListener('click', connectProxy);
    proxyDisconnectBtn.addEventListener('click', disconnectProxy);
    document.getElementById('proxyClear').addEventListener('click', () => proxyLog.innerHTML = '');
    document.getElementById('proxyDownload').addEventListener('click', () => downloadLog(proxyLog, 'proxy.log'));

    llamaConnectBtn.addEventListener('click', connectLlama);
    llamaDisconnectBtn.addEventListener('click', disconnectLlama);
    document.getElementById('llamaClear').addEventListener('click', () => llamaLog.innerHTML = '');
    document.getElementById('llamaDownload').addEventListener('click', () => downloadLog(llamaLog, 'llama-server.log'));

    connectProxy();
    connectLlama();

    window.addEventListener('beforeunload', () => {
      if (esProxy) esProxy.close();
      if (esLlama) esLlama.close();
    });
  </script>
</body>
</html>"""

    # Prepare JSON snapshot for client-side rendering
    # Use shallow copies under locks
    async with srv.counts_lock:
        counts_snapshot = dict(srv.request_counts)
    async with srv.token_lock:
        tokens_snapshot = dict(srv.token_counts)

    model_list = list(srv.config.get("models", {}).keys())
    router_mode = srv.config.get("server", {}).get("llama_router_mode", False)
    router_models = None
    if router_mode:
        router_models = await srv.router_list_models()
    model_list_json = json.dumps(model_list)
    initial_stats_json = json.dumps({"counts": counts_snapshot, "tokens": tokens_snapshot})

    # Replace placeholders with empty containers; client will render using INITIAL_STATS
    html = html.replace('{counts_html}', '')
    html = html.replace('{tokens_html}', '')

    # Inject initial stats and model list script before </body>
    html = html.replace('</body>', f'<script>window.__INITIAL_STATS = {initial_stats_json}; window.__MODEL_LIST = {model_list_json}; window.__ROUTER_MODE = {json.dumps(router_mode)}; window.__ROUTER_MODELS = {json.dumps(router_models)};</script></body>')
    return HTMLResponse(content=html)



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
    
    if model_cfg.get("type") == "local":
        server_config = srv.config.get("server", {})
        router_mode = server_config.get("llama_router_mode", False)
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        if not isinstance(llama_model, str) or not llama_model:
            raise HTTPException(
                status_code=500,
                detail=f"Local model configuration missing llama_model for: {model_name}"
            )
        llama_model_str: str = llama_model

        # If model already active and process running, proceed immediately
        if srv.current_model == llama_model_str and srv.llama_process is not None and (srv.llama_process.poll() is None):
            return await srv.proxy_to_local(request, "v1/embeddings")

        # Try a fast router-mode check: model may already be loaded in router
        if router_mode:
            try:
                if await srv.router_is_model_loaded(llama_model_str):
                    srv.logger.info(f"Router reports model {llama_model_str} already loaded; serving request immediately")
                    srv.current_model = llama_model_str
                    return await srv.proxy_to_local(request, "v1/embeddings")
            except Exception:
                # Non-fatal: fall through to scheduling background load
                srv.logger.debug("Fast router check failed; scheduling background load")

        # Otherwise, schedule a background load and return 503 immediately
        target_model: str = model_name if isinstance(model_name, str) and model_name else llama_model_str
        scheduled = srv.schedule_background_load(target_model)
        srv.logger.info(f"Scheduled background load for embeddings request: {target_model} scheduled={scheduled}")
        return srv._model_loading_response(
            requested_model=model_name if isinstance(model_name, str) else None,
            target_model=target_model,
            scheduled=scheduled,
            endpoint="/v1/embeddings",
        )
    
    elif model_cfg.get("type") == "remote":
        return await srv.proxy_to_remote(request, "v1/embeddings", model_cfg)
    
    raise HTTPException(
        status_code=500,
        detail=f"Invalid model configuration for: {model_name}"
    )



async def proxy_openai_api(request: Request, path: str):
    """
    Main proxy endpoint for OpenAI API requests.
    Routes to local llama-server or remote API based on model.
    """
    # Get the request body to determine the model
    srv = _srv()
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
    if not model_name and srv.current_model:
        model_name = srv.current_model
    
    # Get model configuration
    model_cfg = srv.get_model_config(model_name) if model_name else None
    
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
    
    if model_cfg.get("type") == "local":
        server_config = srv.config.get("server", {})
        router_mode = server_config.get("llama_router_mode", False)
        # Check if we need to switch models
        llama_model = model_cfg.get("llama_model")
        if not isinstance(llama_model, str) or not llama_model:
            raise HTTPException(
                status_code=500,
                detail=f"Local model configuration missing llama_model for: {model_name}"
            )
        llama_model_str: str = llama_model

        # If model already active and process running, proceed immediately
        if srv.current_model == llama_model_str and srv.llama_process is not None and (srv.llama_process.poll() is None):
            return await srv.proxy_to_local(request, f"v1/{path}")

        # Try a fast router-mode check: model may already be loaded in router
        if router_mode:
            try:
                if await srv.router_is_model_loaded(llama_model_str):
                    srv.logger.info(f"Router reports model {llama_model_str} already loaded; serving request immediately")
                    srv.current_model = llama_model_str
                    return await srv.proxy_to_local(request, f"v1/{path}")
            except Exception:
                srv.logger.debug("Fast router check failed; scheduling background load")

        # Otherwise, schedule background load and return 503 so client doesn't hang
        target_model: str = model_name if isinstance(model_name, str) and model_name else llama_model_str
        scheduled = srv.schedule_background_load(target_model)
        srv.logger.info(f"Scheduled background load for request: model={target_model} scheduled={scheduled}")
        return srv._model_loading_response(
            requested_model=model_name if isinstance(model_name, str) else None,
            target_model=target_model,
            scheduled=scheduled,
            endpoint=f"/v1/{path}",
        )
    
    elif model_cfg.get("type") == "remote":
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
    
    if model_cfg.get("type") != "local":
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} is not a local model"
        )
    
    srv.logger.info(f"Admin switch-model requested: {model_name}; current_model before: {current_model}; llama_running: {llama_process is not None and llama_process.poll() is None}")
    if await srv.ensure_model_loaded(model_name):
        srv.logger.info(f"Admin switch-model succeeded: requested={model_name} current_model after: {current_model}")
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






