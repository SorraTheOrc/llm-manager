# Monitoring Infrastructure Deployment Guide

This guide covers deploying Prometheus and Grafana on the LLM proxy stack
host using systemd user services (host-first deployment).

## Overview

- **Prometheus** scrapes the proxy `/metrics` endpoint at `localhost:8000`,
  stores metrics locally, and evaluates alerting rules.
- **Grafana** provides a web UI for visualizing Prometheus metrics using
  pre-configured dashboards imported via provisioning.

## Prerequisites

- LLM proxy is running and exposes `/metrics` on `localhost:8000`.
- `systemctl --user` is available (systemd user services).
- `loginctl enable-linger` has been run for the user so that user services
  survive logout/reboot:

  ```bash
  loginctl enable-linger $USER
  ```

- Ports 9090 (Prometheus), 3000 (Grafana), and 5000 (rocm-exporter) are available.

## Installation

### 1. Download and install Prometheus

```bash
# Set version (update as needed)
PROMETHEUS_VERSION="3.2.1"
ARCH="linux-amd64"

# Download
wget "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"

# Extract
tar xzf "prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"

# Install
mkdir -p ~/bin/prometheus
cp prometheus-${PROMETHEUS_VERSION}.${ARCH}/prometheus ~/bin/prometheus/
cp -r prometheus-${PROMETHEUS_VERSION}.${ARCH}/consoles ~/bin/prometheus/
cp -r prometheus-${PROMETHEUS_VERSION}.${ARCH}/console_libraries ~/bin/prometheus/

# Create data and log directories
mkdir -p ~/.local/share/prometheus/data
mkdir -p ~/.local/state/prometheus/logs

# Clean up
rm -rf "prometheus-${PROMETHEUS_VERSION}.${ARCH}"
rm "prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"
```

### 2. Configure Prometheus

The Prometheus configuration is at `monitoring/prometheus.yml` in this
repository. It is pre-configured to:

- Scrape the proxy `/metrics` endpoint every 15s.
- Load alert rules from `monitoring/llama_memory_alerts.yaml` and
  `monitoring/proxy_5xx_alerts.yaml`.

No additional configuration changes are needed for basic operation.

### 3. Install Prometheus systemd service

Copy the systemd unit and start the service:

```bash
# Install the unit file
mkdir -p ~/.config/systemd/user
cp docs/systemd/prometheus.service ~/.config/systemd/user/prometheus.service

# Reload, enable, and start
systemctl --user daemon-reload
systemctl --user enable prometheus.service
systemctl --user start prometheus.service

# Verify
systemctl --user status prometheus.service
journalctl --user -u prometheus.service -n 20
```

### 4. Download and install Grafana

```bash
# Set version (update as needed)
GRAFANA_VERSION="11.6.0"
ARCH="linux-amd64"

# Download
wget "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.${ARCH}.tar.gz"

# Extract
tar xzf "grafana-${GRAFANA_VERSION}.${ARCH}.tar.gz"

# Install
mkdir -p ~/bin/grafana
cp -r "grafana-${GRAFANA_VERSION}/"* ~/bin/grafana/

# Create data directory
mkdir -p ~/.local/share/grafana/data
mkdir -p ~/.local/state/grafana/logs

# Clean up
rm -rf "grafana-${GRAFANA_VERSION}"
rm "grafana-${GRAFANA_VERSION}.${ARCH}.tar.gz"
```

### 5. Configure Grafana provisioning

Copy the provisioning configuration files and the dashboard JSON:

```bash
# Create provisioning directories
mkdir -p ~/bin/grafana/conf/provisioning/datasources
mkdir -p ~/bin/grafana/conf/provisioning/dashboards

# Copy datasource config
cp monitoring/grafana/datasources/datasources.yaml \
   ~/bin/grafana/conf/provisioning/datasources/

# Copy dashboard provisioning config
cp monitoring/grafana/dashboards/dashboards.yaml \
   ~/bin/grafana/conf/provisioning/dashboards/

# Copy the dashboard JSON to the directory where provisioning expects it
cp monitoring/grafana_llama_memory_dashboard.json \
   ~/bin/grafana/conf/provisioning/dashboards/
```

### 6. Install Grafana systemd service

```bash
# Install the unit file
cp docs/systemd/grafana-server.service ~/.config/systemd/user/grafana-server.service

# Reload, enable, and start
systemctl --user daemon-reload
systemctl --user enable grafana-server.service
systemctl --user start grafana-server.service

# Verify
systemctl --user status grafana-server.service
journalctl --user -u grafana-server.service -n 20
```

## Verification

### Verify Prometheus

```bash
# Check Prometheus targets (should show llama-proxy as UP)
curl http://localhost:9090/api/v1/targets

# Query a metric
curl 'http://localhost:9090/api/v1/query?query=up'

# Check alert rules are loaded
curl http://localhost:9090/api/v1/rules

# Prometheus expression browser
# Open http://localhost:9090 in a browser
```

### Verify Grafana

```bash
# Check Grafana health
curl http://localhost:3000/api/health

# Login and check datasources
# Open http://localhost:3000 in a browser
# Default credentials: admin / admin
# **IMPORTANT**: Change the admin password on first login.

# Verify Prometheus datasource is configured (from CLI)
curl -u admin:admin http://localhost:3000/api/datasources

# Verify dashboards are loaded
curl -u admin:admin http://localhost:3000/api/search?type=dash-db
```

### Verify metrics are flowing

```bash
# Direct proxy /metrics endpoint
curl http://localhost:8000/metrics | head -20

# Query a specific metric in Prometheus
curl 'http://localhost:9090/api/v1/query?query=llama_process_rss_bytes'

# Query the 5xx error counter
curl 'http://localhost:9090/api/v1/query?query=proxy_http_errors_total'
```

## Service Management

```bash
# Start/stop/restart
systemctl --user start prometheus.service
systemctl --user start grafana-server.service

systemctl --user stop prometheus.service
systemctl --user stop grafana-server.service

# Status
systemctl --user status prometheus.service
systemctl --user status grafana-server.service

# View logs (follow mode)
journalctl --user -u prometheus.service -f
journalctl --user -u grafana-server.service -f

# Enable at boot (requires loginctl enable-linger)
systemctl --user enable prometheus.service
systemctl --user enable grafana-server.service
```

## Configuration Reference

### Prometheus (`monitoring/prometheus.yml`)

| Setting | Value | Description |
|---------|-------|-------------|
| `scrape_interval` | 15s | How often to scrape targets |
| `evaluation_interval` | 15s | How often to evaluate alert rules |
| Scrape target | `localhost:8000` | Proxy `/metrics` endpoint |
| Alert rules | `monitoring/llama_memory_alerts.yaml`, `monitoring/proxy_5xx_alerts.yaml` | Pre-defined alert rules |

### Grafana Datasource (`monitoring/grafana/datasources/datasources.yaml`)

| Setting | Value | Description |
|---------|-------|-------------|
| `name` | Prometheus | Datasource name |
| `type` | prometheus | Datasource type |
| `url` | `http://localhost:9090` | Prometheus API URL |
| `isDefault` | true | Set as default datasource |

### Grafana Dashboard (`monitoring/grafana_llama_memory_dashboard.json`)

- **Title**: LLama Server Memory
- **Panels**:
  - llama-server RSS (bytes) — time-series graph
  - Models loaded — gauge/time-series
  - Proxy 5xx Errors — rate graph
  - GPU VRAM Usage (bytes) — time-series graph showing total and used VRAM
- **Datasource**: Prometheus

## ROCm Exporter

The AMD ROCm exporter (`rocm-exporter`) provides GPU VRAM metrics for
Prometheus scraping. It is deployed as a systemd user service on the host.

### Prerequisites

- ROCm is installed and GPU is accessible:
  ```bash
  rocm-smi --showtag
  rocm-smi --showmeminfo vram
  ```
- Port 5000 is available for the exporter's metrics endpoint.

### Installation

See `docs/systemd/rocm-exporter.service` for full installation instructions,
including both user-service and system-service deployment options.

Quick-start (user service):

```bash
# Download and install the rocm-exporter binary
# See https://github.com/amd/rocm-exporter/releases for the latest version
ROCM_EXPORTER_VERSION="0.1.0"
wget "https://github.com/amd/rocm-exporter/releases/download/v${ROCM_EXPORTER_VERSION}/rocm-exporter-${ROCM_EXPORTER_VERSION}.linux-amd64.tar.gz"
tar xzf rocm-exporter-${ROCM_EXPORTER_VERSION}.linux-amd64.tar.gz
mkdir -p ~/bin/rocm-exporter
cp rocm-exporter ~/bin/rocm-exporter/
rm -rf rocm-exporter rocm-exporter-${ROCM_EXPORTER_VERSION}.linux-amd64.tar.gz

# Create log directory
mkdir -p ~/.local/state/rocm-exporter/logs

# Install the systemd unit
mkdir -p ~/.config/systemd/user
cp docs/systemd/rocm-exporter.service ~/.config/systemd/user/rocm-exporter.service
systemctl --user daemon-reload
systemctl --user enable rocm-exporter.service
systemctl --user start rocm-exporter.service

# Verify
systemctl --user status rocm-exporter.service
```

### Verification

```bash
# Check the metrics endpoint
curl http://localhost:5000/metrics | grep rocm_vram

# Expected metrics:
#   rocm_vram_total_bytes{gpu_id="0"} 17179869184
#   rocm_vram_used_bytes{gpu_id="0"}  8589934592
#   rocm_vram_free_bytes{gpu_id="0"} 8589934592
```

### Prometheus scrape configuration

The rocm-exporter is already configured as a scrape target in
`monitoring/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "rocm-exporter"
    static_configs:
      - targets: ["localhost:5000"]
    metrics_path: /metrics
```

No additional configuration changes are needed for basic operation.

## Alerting Rules

The following alert rules are loaded into Prometheus:

| Alert Name | Severity | Condition | Summary |
|------------|----------|-----------|---------|
| `LlamaMemoryHighWarning` | warning | RSS > 75% of 90GB for 2m | Process memory above 75% threshold |
| `LlamaMemoryHighCritical` | critical | RSS > 90GB for 1m | Process memory above 90GB |
| `ProxyHttpErrorsHigh` | critical | 5xx rate > 5/s for 5m | Proxy returning excessive errors |
| `GpuVramHighWarning` | warning | VRAM usage > 75% for 2m | GPU VRAM above 75% threshold |
| `GpuVramHighCritical` | critical | VRAM usage > 90% for 1m | GPU VRAM above 90% threshold |

Alerts are visible in the Prometheus UI (`http://localhost:9090/alerts`) and
API (`http://localhost:9090/api/v1/rules`). Alert notification routing
(e.g., email, webhook) requires Alertmanager deployment, which is a separate
future enhancement.

## Troubleshooting

### Prometheus targets show "DOWN"

- Verify the proxy is running: `curl http://localhost:8000/health`
- Check Prometheus logs: `journalctl --user -u prometheus.service -n 50`
- Verify the proxy `/metrics` endpoint: `curl http://localhost:8000/metrics`

### Grafana fails to start

- Check Grafana logs: `journalctl --user -u grafana-server.service -n 50`
- Verify port 3000 is available: `ss -tlnp | grep 3000`
- Ensure the provisioning directories exist and files have correct permissions

### Dashboards not showing in Grafana

- Verify provisioning configuration files are in the correct directory:
  `ls ~/bin/grafana/conf/provisioning/datasources/`
  `ls ~/bin/grafana/conf/provisioning/dashboards/`
- Check Grafana logs for provisioning errors
- The dashboard JSON file must be in the directory specified by the
  provisioning config's `options.path`

## Security Notes

- **Default credentials**: Grafana starts with `admin`/`admin`. Change the
  password immediately on first login.
- **Prometheus API**: Prometheus does not have built-in authentication.
  Consider adding a reverse proxy with auth if the host is exposed to a
  network.
- **Systemd user services**: These run under the current user's session.
  Use `loginctl enable-linger` to ensure services persist across reboots.
