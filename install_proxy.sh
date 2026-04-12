#!/bin/bash
#
# Install script for LLama Proxy Server
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_DIR="$SCRIPT_DIR/proxy"
VENV_DIR="$PROXY_DIR/.venv"
LOG_DIR="$PROXY_DIR/logs"

echo "=========================================="
echo "LLama Proxy Server Installation"
echo "=========================================="
echo

# Keep setup minimal for development and ad-hoc starts; do not install
# systemd unit files from the repository. If you wish to run the proxy as
# a service, create and manage service unit files outside this repo.
echo

# Create virtual environment
echo "[1/5] Creating Python virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "      Created: $VENV_DIR"
else
    echo "      Already exists: $VENV_DIR"
fi
echo

# Activate and install dependencies
echo "[2/5] Installing Python dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip > /dev/null
pip install -r "$PROXY_DIR/requirements.txt"
echo

echo "[3/5] Setting up log directory..."
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR" || true
echo "      Logs will be written to: $LOG_DIR"
echo "      To use a system log directory instead (optional):"
echo "        sudo mkdir -p /var/log/llama-proxy && sudo chown \$USER:\$USER /var/log/llama-proxy"
echo

# Final instructions
echo "[5/5] Setup complete!"
echo
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo
echo "1. Edit the configuration file:"
echo "   $PROXY_DIR/config.yaml"
echo
echo "2. Set API keys for remote services (if needed):"
echo "   export OPENAI_API_KEY='your-key'"
echo "   export ANTHROPIC_API_KEY='your-key'"
echo
echo "3. If you run a systemd unit you created separately, add API keys to that unit using:"
echo "   sudo systemctl edit <your-unit-name>"
echo "   Add: Environment=\"OPENAI_API_KEY=your-key\""
echo
echo "=========================================="
echo "Running the Server"
echo "=========================================="
echo
echo "Manual start (development):"
echo "   cd $PROXY_DIR"
echo "   source .venv/bin/activate"
echo "   python -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000"
echo
echo "Service start (production):"
echo "   If you installed a service unit outside this repo, start it with your service manager (example: systemd):"
echo "     sudo systemctl start <your-unit-name>"
echo
# Get the machine's IP address (first non-loopback IPv4 address)
MACHINE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [[ -z "$MACHINE_IP" ]]; then
    MACHINE_IP=$(ip route get 1 2>/dev/null | awk '{print $7; exit}')
fi
if [[ -z "$MACHINE_IP" ]]; then
    MACHINE_IP="localhost"
fi

echo "=========================================="
echo "API Endpoints"
echo "=========================================="
echo
echo "  Health check:     GET  http://${MACHINE_IP}:8000/health"
echo "  List models:      GET  http://${MACHINE_IP}:8000/v1/models"
echo "  Chat completions: POST http://${MACHINE_IP}:8000/v1/chat/completions"
echo "  Completions:      POST http://${MACHINE_IP}:8000/v1/completions"
echo
echo "  Admin endpoints:"
echo "    Reload config:  POST http://${MACHINE_IP}:8000/admin/reload-config"
echo "    Switch model:   POST http://${MACHINE_IP}:8000/admin/switch-model/{model}"
echo "    Stop server:    POST http://${MACHINE_IP}:8000/admin/stop-server"
echo
