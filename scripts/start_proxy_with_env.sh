#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# start_proxy_with_env.sh
#
# Starts the LLM proxy with required API keys resolved from:
#   1. Environment variables (already set)
#   2. Pi's auth.json (~/.pi/agent/auth.json) as fallback
#
# Discovers all required `api_key_env` values from proxy/config.yaml
# and ensures each is available before starting the server.
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_DIR="$(cd "$SCRIPT_DIR/../proxy" && pwd)"
CONFIG_FILE="$PROXY_DIR/config.yaml"
AUTH_FILE="$HOME/.pi/agent/auth.json"

# ---- Resolve API keys from config.yaml ---------------------------------

resolve_api_keys() {
  local missing=()

  # Extract all unique api_key_env values from config.yaml
  while IFS='' read -r env_var; do
    [[ -z "$env_var" ]] && continue

    # Already set in environment — nothing to do
    if [[ -n "${!env_var:-}" ]]; then
      echo "[env] $env_var already set from environment"
      continue
    fi

    # Try to resolve from pi's auth.json
    if [[ -f "$AUTH_FILE" ]]; then
      resolved="$(resolve_from_auth_json "$env_var")"
      if [[ -n "$resolved" ]]; then
        export "$env_var=$resolved"
        echo "[env] $env_var resolved from ~/.pi/agent/auth.json"
        continue
      fi
    fi

    # Not found anywhere — annotate with which model(s) need it
    local models_using
    models_using="$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
models = []
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        if p.get('api_key_env') == '$env_var':
            models.append(name)
print(', '.join(models))
" 2>/dev/null || echo 'unknown')"
    missing+=("$env_var  (required by: $models_using)")
  done < <(python3 -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
keys = set()
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        env_key = p.get('api_key_env')
        if env_key:
            keys.add(env_key)
for k in sorted(keys):
    print(k)
")

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: The following API key environment variables are not set"
    echo "       and could not be resolved from ~/.pi/agent/auth.json:"
    for key in "${missing[@]}"; do
      echo "  - $key"
    done
    echo ""
    echo "Set each as an environment variable before starting the proxy, for example:"
    echo "  export GITHUB_TOKEN=ghp_..."
    echo "  export OPENCODE_API_KEY=sk-..."
    echo ""
    echo "Or add the key to $AUTH_FILE under the matching provider name"
    echo "(e.g. OPENCODE_API_KEY -> auth.json[\"opencode\"] or auth.json[\"opencode-go\"])"
    return 1
  fi
}

# ---- Map api_key_env name to auth.json key ------------------------------
# Known mappings from config.yaml api_key_env -> auth.json provider key.
# If a direct match is not found, tries heuristics:
#   - lowercase the env var name, strip _API_KEY suffix
#   - e.g. OPENCODE_API_KEY -> auth.json["opencode"]
resolve_from_auth_json() {
  local env_var="$1"
  local value

  # Direct match: auth.json key == env var name (lowercased)
  value="$(python3 -c "
import json, sys
try:
    with open('$AUTH_FILE') as f:
        auth = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)

key = '$env_var'.lower()
# Try exact lowercase match first
if key in auth and auth[key].get('type') == 'api_key':
    print(auth[key]['key'])
    sys.exit(0)

# Try stripping _API_KEY suffix
if key.endswith('_api_key'):
    stem = key[:-8]  # remove '_api_key'
    if stem in auth and auth[stem].get('type') == 'api_key':
        print(auth[stem]['key'])
        sys.exit(0)

# Try uppercase regex-style: GITHUB_TOKEN -> github-token or github
for alt in (key.replace('_', '-'), key.split('_')[0].lower()):
    if alt in auth and auth[alt].get('type') == 'api_key':
        print(auth[alt]['key'])
        sys.exit(0)

sys.exit(1)
")" && echo "$value" || true

  if [[ -n "$value" ]]; then
    echo "$value"
  fi
}

# ---- Main ---------------------------------------------------------------

echo "=== LLM Proxy: Resolving API keys ==="
resolve_api_keys

echo ""
echo "=== Starting proxy server ==="
export PYTHONPATH="$PROXY_DIR"
exec "$PROXY_DIR/../.venv/bin/python3" -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000
