#!/bin/bash
#
# Install script for LLama Proxy Server
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SERVICE_FILE="$SCRIPT_DIR/llama-proxy.service"
SYSTEMD_DIR="/etc/systemd/system"
LOG_DIR="/var/log/llama-proxy"

echo "=========================================="
echo "LLama Proxy Server Installation"
echo "=========================================="
echo

# Check if running as root for systemd installation
INSTALL_SYSTEMD=false
if [[ $EUID -eq 0 ]]; then
    INSTALL_SYSTEMD=true
    echo "[INFO] Running as root - will install systemd service"
else
    echo "[INFO] Not running as root - systemd service will need manual installation"
    echo "[INFO] Run with sudo to install systemd service automatically"
fi
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
pip install -r "$SCRIPT_DIR/requirements.txt"
echo

# Create log directory
echo "[3/5] Setting up log directory..."
if [[ $INSTALL_SYSTEMD == true ]]; then
    mkdir -p "$LOG_DIR"
    chown "$SUDO_USER:$SUDO_USER" "$LOG_DIR" 2>/dev/null || true
    chmod 755 "$LOG_DIR"
    echo "      Created: $LOG_DIR"
else
    echo "      [SKIP] Run as root to create $LOG_DIR"
    echo "      In development mode, logs will be written to: $SCRIPT_DIR/logs/"
    echo "      Or create manually: sudo mkdir -p $LOG_DIR && sudo chown \$USER:\$USER $LOG_DIR"
fi
echo

# Install systemd service
echo "[4/5] Installing systemd service..."
if [[ $INSTALL_SYSTEMD == true ]]; then
    # Update service file with correct user and UID
    CURRENT_USER="${SUDO_USER:-$USER}"
    CURRENT_UID=$(id -u "$CURRENT_USER")
    CURRENT_HOME=$(eval echo "~$CURRENT_USER")
    
    # Replace user/group and paths with actual values
    sed -e "s/User=rgardler/User=$CURRENT_USER/g" \
        -e "s/Group=rgardler/Group=$CURRENT_USER/g" \
        -e "s|/run/user/1000|/run/user/$CURRENT_UID|g" \
        -e "s|/home/rgardler|$CURRENT_HOME|g" \
        "$SERVICE_FILE" > "$SYSTEMD_DIR/llama-proxy.service"
    
    systemctl daemon-reload
    echo "      Installed: $SYSTEMD_DIR/llama-proxy.service"
    echo
    echo "      To enable on boot: sudo systemctl enable llama-proxy"
    echo "      To start now:      sudo systemctl start llama-proxy"
    echo "      To check status:   sudo systemctl status llama-proxy"
else
    echo "      [SKIP] Run as root to install systemd service"
    echo "      Manual install:"
    echo "        sudo cp $SERVICE_FILE $SYSTEMD_DIR/"
    echo "        sudo systemctl daemon-reload"
    echo "        sudo systemctl enable llama-proxy"
fi
echo

# Final instructions
echo "[5/5] Setup complete!"
echo
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo
echo "1. Edit the configuration file:"
echo "   $SCRIPT_DIR/config.yaml"
echo
echo "2. Set API keys for remote services (if needed):"
echo "   export OPENAI_API_KEY='your-key'"
echo "   export ANTHROPIC_API_KEY='your-key'"
echo
echo "3. For systemd, add API keys to the service file:"
echo "   sudo systemctl edit llama-proxy"
echo "   Add: Environment=\"OPENAI_API_KEY=your-key\""
echo
echo "=========================================="
echo "Running the Server"
echo "=========================================="
echo
echo "Manual start (development):"
echo "   cd $SCRIPT_DIR"
echo "   source .venv/bin/activate"
echo "   python -m uvicorn server:app --host 0.0.0.0 --port 8000"
echo
echo "Systemd start (production):"
echo "   sudo systemctl start llama-proxy"
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
