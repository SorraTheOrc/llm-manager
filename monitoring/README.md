# Monitoring stack: deployed state, deployment and queries

This is the deployment guide for the monitoring stack that runs **on this
host**. It records what is actually running (versions, binds, exposure), how to
reproduce it on a fresh host, how to verify it, and how to query it.

It supersedes the earlier version of this guide, which documented a stack that
could not have been deployed (it installed a `rocm-exporter` release that does
not exist and copied Prometheus 2.x `consoles/` directories that Prometheus 3.x
no longer ships).

## What runs here

Everything is a **systemd user service** (host-first, no containers), kept
alive across reboots with `loginctl enable-linger`.

| Unit | Version | Listens on | Scraped / exposed | Documentation |
|---|---|---|---|---|
| `prometheus.service` | 3.15.1 | `127.0.0.1:9090`, `100.79.231.101:9090`, `[fd7a:115c:a1e0::9b33:e765]:9090` | tailnet UI/API, 90 d retention | [docs/systemd/prometheus.service](../docs/systemd/prometheus.service) |
| `node_exporter.service` | 1.12.1 | `127.0.0.1:9100` | scraped by Prometheus | [docs/systemd/node_exporter.service](../docs/systemd/node_exporter.service) |
| `gpu-metrics.service` | n/a (bash sampler) | — (writes `gpu.prom`) | 10 s sysfs sampler for node_exporter's textfile collector | [docs/systemd/gpu-metrics.service](../docs/systemd/gpu-metrics.service) |
| `rocm-exporter.service` | — | — | **not installable**: no published release artefacts | [docs/systemd/rocm-exporter.service](../docs/systemd/rocm-exporter.service) |
| `grafana-server.service` | 11.6.0 (documented) | `:3000` if deployed | optional dashboards; **not running on this host** | [docs/systemd/grafana-server.service](../docs/systemd/grafana-server.service) |

Scrape pipeline:

```
llama-proxy :8000/metrics ─┐
node_exporter :9100/metrics ┤─► prometheus :9090 (90 d) ─► (optional) grafana :3000
  ├ hwmon collector  ◄── /sys/class/hwmon  (GPU power)
  └ textfile collector ◄── gpu.prom ◄── gpu-metrics.service (10 s sysfs sampler)
```

## Prerequisites

- The LLM proxy is running and exposes `/metrics` on `localhost:8000`
  (`systemctl --user status llama-proxy.service`).
- `systemctl --user` is available and linger is enabled, so user services
  survive logout and start at boot:

  ```bash
  loginctl enable-linger "$USER"
  loginctl show-user "$USER" -p Linger     # expected: Linger=yes
  ```

- Tailscale is up on this node (Prometheus binds tailnet addresses):

  ```bash
  tailscale status | head -3
  tailscale ip -4     # expected: 100.79.231.101 on this host
  tailscale ip -6     # expected: fd7a:115c:a1e0::9b33:e765
  ```

- Ports 9090 (Prometheus), 9100 (node_exporter) and, if Grafana is deployed,
  3000, are free.

## Exposure boundary

**Decision: Prometheus is exposed on loopback and the tailnet only. It is
deliberately not exposed to the LAN, and it has no authentication in front of
it.**

Reasoning:

- Prometheus has **no built-in authentication or authorisation**. Whatever can
  reach the listener can read every metric, and can `POST /-/quit` to stop the
  server. Any exposure therefore has to be network-level, not application-level.
- The **tailnet is device-authenticated** (WireGuard mesh: a peer must be a
  device admitted to the tailnet), so it is an acceptable boundary for an
  unauthenticated service.
- The LAN (`wlp195s0`, `192.168.0.199/24` here) is **not** an acceptable
  boundary: it contains devices this stack does not control, so the wildcard
  bind that this guide previously documented (`--web.listen-address=:9090`) was
  a real exposure bug, not a convenience.

What that means in practice — anything on the tailnet can, without any
credential:

| Endpoint | Effect |
|---|---|
| `/`, `/graph`, `/alerts`, `/targets` | full UI, query execution |
| `/api/v1/*` | full query/scrape/config read API |
| `/-/healthy`, `/-/ready` | health probes |
| `POST /-/quit` | **stops the Prometheus process** (it has no enabling flag, so it is always registered; `Restart=always` brings it back) |
| `POST /-/reload` | answered with *"Lifecycle API is not enabled."* — `--web.enable-lifecycle` is not passed |
| `/api/v1/admin/tsdb/delete_series` | answered with *"admin APIs disabled"* — `--web.enable-admin-api` is not passed |

The last two are disabled on purpose. Enabling either would put configuration
reloads and irreversible time-series deletion on the tailnet with no
authentication; they are single-operator actions, done with
`systemctl --user restart prometheus.service` instead.

### Verification of the boundary

```bash
# Tailnet paths answer
for u in http://127.0.0.1:9090 http://100.79.231.101:9090 \
         'http://[fd7a:115c:a1e0::9b33:e765]:9090' \
         http://sorra-agents-1.tail37f0c2.ts.net:9090; do
  printf '%-45s %s\n' "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$u/-/ready")"
done
# expected: 200 200 200 200

# The LAN address is refused
curl -s -o /dev/null -w '%{http_code}\n' --max-time 4 http://192.168.0.199:9090/-/ready
# expected: 000 (connection refused)

# Only the intended sockets listen
ss -tln | grep ':9090'
# expected: 127.0.0.1:9090, 100.79.231.101:9090, [fd7a:115c:a1e0::9b33:e765]:9090

# Lifecycle and admin APIs are disabled
curl -s -X POST http://127.0.0.1:9090/-/reload                      # Lifecycle API is not enabled.
curl -s -X POST http://127.0.0.1:9090/api/v1/admin/tsdb/delete_series
# {"status":"error","errorType":"unavailable","error":"admin APIs disabled"}
```

### Changing the boundary

Do not edit `~/.config/systemd/user/prometheus.service` by hand — the
repository would drift from the running state. Edit
[docs/systemd/prometheus.service](../docs/systemd/prometheus.service), re-run
the extraction and restart:

- **Widen to the LAN**: add or change a `--web.listen-address` to the LAN
  interface. Treat that as a security change: unauthenticated Prometheus with
  `/-/quit` reachable from every device on the network.
- **Node re-registered in Tailscale**: the tailnet IPv4/IPv6 change, so both the
  listen addresses *and* `--web.external-url` must be updated together (see
  [docs/systemd/prometheus.service § If this node is re-registered in Tailscale](../docs/systemd/prometheus.service#if-this-node-is-re-registered-in-tailscale)).

## 1. Install Prometheus

```bash
# Set version (this host runs 3.15.1)
PROMETHEUS_VERSION="3.15.1"
ARCH="linux-amd64"

wget "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"
tar xzf "prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"

# Install the binaries only: Prometheus 3.x ships no consoles/ or
# console_libraries/, so copying them (as this guide used to instruct) fails.
mkdir -p ~/bin/prometheus
cp "prometheus-${PROMETHEUS_VERSION}.${ARCH}/prometheus" \
   "prometheus-${PROMETHEUS_VERSION}.${ARCH}/promtool" ~/bin/prometheus/

# Data, logs and configuration directories
mkdir -p ~/.local/share/prometheus/data
mkdir -p ~/.local/state/prometheus/logs

rm -rf "prometheus-${PROMETHEUS_VERSION}.${ARCH}" \
       "prometheus-${PROMETHEUS_VERSION}.${ARCH}.tar.gz"

# Sanity check the binary
~/bin/prometheus/prometheus --version   # expected: prometheus, version 3.15.1
```

Install the systemd unit **from the documented unit**, not with `cp`:
`docs/systemd/*.service` are documents that wrap the unit in an `ini` block, so
copying one installs markdown that systemd cannot parse. This helper extracts
the block and is used for every unit in this guide:

```bash
# Install a unit file from the docs in this repository.
unit_from_doc() {
  unit="$1"
  mkdir -p ~/.config/systemd/user
  awk '/^```ini$/{f=1;next} /^```$/{f=0} f' "docs/systemd/${unit}.service" \
    > "$HOME/.config/systemd/user/${unit}.service"
}
```

```bash
cd /path/to/llm          # repository root, so docs/systemd/ resolves
unit_from_doc prometheus
cat ~/.config/systemd/user/prometheus.service   # must be a plain unit, no markdown

systemctl --user daemon-reload
systemctl --user enable --now prometheus.service
systemctl --user status prometheus.service
journalctl --user -u prometheus.service -n 20
```

If the service restart-loops with `cannot assign requested address`, the tailnet
addresses do not exist yet: bring Tailscale up, then
`systemctl --user restart prometheus.service` (the unit retries every 10 s on
its own, so this is only needed to skip the wait).

## 2. Configure Prometheus

The configuration lives at `monitoring/prometheus.yml` in this repository and is
read in place — the unit points `--config.file` at it, so there is nothing to
copy. It scrapes:

- `llama-proxy` — `localhost:8000/metrics`, every 15 s;
- `node-exporter` — `localhost:9100/metrics`, every 15 s (host metrics, GPU
  power, and the GPU memory textfile series).

and evaluates the alert rules in `monitoring/*.yaml`.

> GPU utilisation is derived from **power draw**
> (`node_hwmon_power_average_watt`), not `gpu_busy_percent`: on this Strix Halo
> APU that counter is pinned at 100% regardless of GPU state
> (LP-0MS9GA6QL007XTBP). GPU memory pressure is measured on **GTT**, not the
> 1 GiB VRAM carve-out. See
> [gpu_compute_rules.yaml](gpu_compute_rules.yaml) and
> [gpu_memory_alerts.yaml](gpu_memory_alerts.yaml) for the reasoning.

### Rule paths are relative to the config file

`rule_files` entries are resolved **relative to the directory of the config file
that loads them**, not to the working directory and not to the repository root.
So in `monitoring/prometheus.yml` they must be **bare file names**:

```yaml
rule_files:
  - "gpu_compute_rules.yaml"
  - "gpu_memory_alerts.yaml"
```

Listing them as `monitoring/gpu_compute_rules.yaml` makes Prometheus look for
`monitoring/monitoring/gpu_compute_rules.yaml` and refuse to start. This costs a
failed start every time it is reintroduced, which is why
`proxy/tests/test_monitoring_deployment.py` (a) forbids the `monitoring/` prefix,
(b) asks `promtool` how many rule files it actually loaded and compares that with
the config, and (c) fails if a rule file on disk is not loaded — `promtool check
config` alone exits 0 even when a `rule_files` glob matches nothing.

Validate before restarting:

```bash
~/bin/prometheus/promtool check config monitoring/prometheus.yml
# expected: SUCCESS: 5 rule files found
#           SUCCESS: monitoring/prometheus.yml is valid prometheus config file syntax
~/bin/prometheus/promtool test rules monitoring/tests/gpu_rules_test.yml
# expected: SUCCESS
systemctl --user restart prometheus.service
```

## 3. Host and GPU metrics: node_exporter + gpu-metrics

Both units are needed for full GPU coverage, and both are loopback-only:

- **node_exporter 1.12.1** (`127.0.0.1:9100`) provides host metrics plus GPU
  power through `--collector.hwmon`, and serves the GPU memory textfile through
  `--collector.textfile.directory`.
- **gpu-metrics.service** is a 10 s bash loop that samples
  `/sys/class/drm/card0/device/mem_info_*` and `gpu_busy_percent` into
  `~/.local/share/node_exporter/textfile/gpu.prom`.

Full install steps, metric names and troubleshooting:
[docs/systemd/node_exporter.service](../docs/systemd/node_exporter.service) and
[docs/systemd/gpu-metrics.service](../docs/systemd/gpu-metrics.service).
Short form:

```bash
mkdir -p ~/.local/share/node_exporter/textfile ~/.local/state/node_exporter/logs
mkdir -p ~/.local/bin
# install ~/.local/bin/gpu-metrics-textfile.sh (see docs/systemd/gpu-metrics.service)
chmod +x ~/.local/bin/gpu-metrics-textfile.sh

cd /path/to/llm
unit_from_doc node_exporter
awk '/^```ini$/{f=1;next} /^```$/{f=0} f' docs/systemd/gpu-metrics.service \
  > ~/.config/systemd/user/gpu-metrics.service

systemctl --user daemon-reload
systemctl --user enable --now node_exporter.service gpu-metrics.service
```

Resulting GPU series: `gpu_gtt_used_bytes`, `gpu_vram_used_bytes`,
`gpu_vram_total_bytes`, `gpu_busy_percent` (pinned at 100% on this APU — not a
utilisation signal), plus `node_hwmon_power_average_watt{chip_name="amdgpu"}`.

## 4. Grafana (optional, not deployed here)

Grafana is **not running on this host** and is not part of the verified stack.
The dashboard and provisioning files are in the repository
(`monitoring/grafana/`, `monitoring/grafana_llama_memory_dashboard.json`) and
the unit is documented in
[docs/systemd/grafana-server.service](../docs/systemd/grafana-server.service),
but treat deploying it as a separate change with its own review: Grafana binds
`:3000` (wildcard, i.e. LAN-reachable) by default and ships `admin`/`admin`
credentials, which is a wider exposure than the tailnet-only boundary Prometheus
uses. If you deploy it, bind it to loopback and the tailnet addresses the same
way as Prometheus and change the admin password first.

```bash
# Sketch only — see docs/systemd/grafana-server.service for the full steps
cp monitoring/grafana/datasources/datasources.yaml ~/bin/grafana/conf/provisioning/datasources/
cp monitoring/grafana/dashboards/dashboards.yaml   ~/bin/grafana/conf/provisioning/dashboards/
cp monitoring/grafana_llama_memory_dashboard.json  ~/bin/grafana/conf/provisioning/dashboards/
```

Known gap: the "GPU VRAM Usage" panel in
`monitoring/grafana_llama_memory_dashboard.json` still queries `rocm_vram_*`,
which nothing publishes. It needs repointing at `gpu_gtt_used_bytes` /
`gpu_vram_used_bytes` before Grafana is deployed. Prometheus itself is
unaffected.

## 5. Verification (fresh-host checklist)

Run in order; every command states its expected result.

```bash
# 1. All three units active
systemctl --user is-active prometheus.service node_exporter.service gpu-metrics.service
# expected: active (three lines)

# 2. Prometheus ready on loopback and tailnet, refused on the LAN
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:9090/-/ready            # 200
curl -s -o /dev/null -w '%{http_code}\n' http://100.79.231.101:9090/-/ready       # 200
curl -s -o /dev/null -w '%{http_code}\n' --max-time 4 http://192.168.0.199:9090/-/ready  # 000

# 3. Effective flags: binds, external URL, retention, admin APIs off
curl -s http://127.0.0.1:9090/api/v1/status/flags | jq -r '.data | .["storage.tsdb.retention.time"], .["web.listen-address"], .["web.external-url"], .["web.enable-admin-api"]'
# expected: 90d
#           127.0.0.1:9090,100.79.231.101:9090,[fd7a:115c:a1e0::9b33:e765]:9090
#           http://100.79.231.101:9090/
#           false

# 4. Both scrape targets up
curl -s 'http://127.0.0.1:9090/api/v1/targets?state=active' \
  | jq -r '.data.activeTargets[] | "\(.labels.job) \(.health) \(.scrapeUrl)"'
# expected: llama-proxy up http://localhost:8000/metrics
#           node-exporter up http://localhost:9100/metrics

# 5. Rules loaded: 5 groups, 10 rules
curl -s http://127.0.0.1:9090/api/v1/rules | jq -r '.data.groups[] | "\(.name) \(.rules|length)"'
# expected: gpu-compute.rules 4
#           gpu-memory.rules 2
#           proxy-5xx.rules 1
#           llama-remote-stream.rules 1
#           llama-slots.rules 2

# 6. GPU series are present and sane
curl -s --get http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=gpu_gtt_used_bytes'
# expected: one series, value in the tens of GiB when models are loaded

# 7. Config and rules validate
~/bin/prometheus/promtool check config monitoring/prometheus.yml
```

## Query cookbook

All queries below are PromQL for the Prometheus expression browser
(`http://100.79.231.101:9090/graph`) or the HTTP API:

```bash
curl -s --get http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=<PROMQL>'
```

### GPU computing fraction (from power)

GPU utilisation is a **power** measurement on this hardware. `gpu:computing:bool`
is a recording rule that is 1 while the GPU draws at or above the 90 W compute
threshold and 0 otherwise (`bool` is essential: without it idle samples are
filtered out and averages overstate utilisation).

```promql
# Computing fraction over the last 2 hours (0..1)
avg_over_time(gpu:computing:bool[2h])

# Computing fraction over the last 24 hours
avg_over_time(gpu:computing:bool[24h])

# Total computing seconds in the last 24 hours
sum_over_time(gpu:computing:bool[24h:15s]) * 15

# Smoothed power draw for trend graphs (idle ≈ 40-54 W, computing ≈ 115 W)
gpu:power_watts:avg5m

# Raw power series behind the rules
gpu:power_watts:amdgpu
```

Sanity note: the sample values depend on the machine's state. Verified
2026-10-25 on this host: `avg_over_time(gpu:computing:bool[2h])` → 0.5,
`sum_over_time(...[24h:15s]) * 15` → 4605 s, `gpu:power_watts:avg5m` → 67.7 W.

### GPU memory: GTT and VRAM

Models live in **GTT** (system RAM mapped for the GPU), not in the 1 GiB VRAM
carve-out, so GTT is the number that matters.

```promql
# GTT in use (bytes) — the pool models map into
gpu_gtt_used_bytes

# GTT as a fraction of the 128 GiB pool (the alert thresholds are 0.75 / 0.90)
gpu_gtt_used_bytes / 137438953472

# GTT and VRAM side by side, written in GiB
gpu_gtt_used_bytes / 1024^3
gpu_vram_used_bytes / 1024^3
gpu_vram_total_bytes / 1024^3

# GTT peak and growth over a day
max_over_time(gpu_gtt_used_bytes[24h])
delta(gpu_gtt_used_bytes[1h])
```

### Host metrics

```promql
# Load average, 1 minute
node_load1

# Available memory in GiB
node_memory_MemAvailable_bytes / 1024^3

# Free space on / in GiB
node_filesystem_avail_bytes{mountpoint="/"} / 1024^3

# CPU utilisation fraction per mode (idle excluded)
1 - rate(node_cpu_seconds_total{mode="idle"}[5m])

# Proxy 5xx rate (the proxy's own counter)
rate(proxy_http_errors_total{status="5xx"}[5m])

# Scrape health for both jobs
up
```

### Retention window

Configured retention is 90 days (`--storage.tsdb.retention.time=90d`; the
`prometheus_tsdb_*` self-metrics are not scraped by this instance, so read it
from the flags API). Actual retained history is bounded by the oldest sample:

```bash
curl -s http://127.0.0.1:9090/api/v1/status/flags \
  | jq -r '.data["storage.tsdb.retention.time"]'
# expected: 90d
```

```promql
# Oldest sample held by the TSDB, as an epoch timestamp
min(timestamp(up))

# How much history is actually retained, in days
(time() - min(timestamp(up))) / 86400
```

## Alerting rules

| Alert | Severity | Condition | For | Source |
|---|---|---|---|---|
| `GpuComputingSustained` | warning | `gpu:computing:bool == 1` (GPU at/above 90 W continuously) | 15m | [gpu_compute_rules.yaml](gpu_compute_rules.yaml) |
| `GpuGttHighWarning` | warning | `gpu_gtt_used_bytes > 0.75 * 128 GiB` | 2m | [gpu_memory_alerts.yaml](gpu_memory_alerts.yaml) |
| `GpuGttHighCritical` | critical | `gpu_gtt_used_bytes > 0.90 * 128 GiB` | 1m | [gpu_memory_alerts.yaml](gpu_memory_alerts.yaml) |
| `ProxyHttpErrorsHigh` | critical | `rate(proxy_http_errors_total{endpoint="/v1/chat/completions",status="5xx"}[5m]) > 5` | 5m | [proxy_5xx_alerts.yaml](proxy_5xx_alerts.yaml) |
| `LlamaSlotsQueryFailuresHigh` | warning | `rate(llama_slots_query_failures_total[5m]) > 1` | 5m | [slots_query_alerts.yaml](slots_query_alerts.yaml) |
| `LlamaSlotsQueryFailuresCritical` | critical | `rate(llama_slots_query_failures_total[5m]) > 10` | 10m | [slots_query_alerts.yaml](slots_query_alerts.yaml) |
| `RemoteStreamWatchdogTerminated` | warning | `rate(llama_remote_stream_terminated_total[5m]) > 0` | 5m | [remote_stream_alerts.yaml](remote_stream_alerts.yaml) |

There are also three **recording** rules in `gpu_compute_rules.yaml`:
`gpu:power_watts:amdgpu`, `gpu:computing:bool`, `gpu:power_watts:avg5m`.

Alerts are visible in the UI (`/alerts`) and API (`/api/v1/rules`). Notification
routing (email, webhook) needs Alertmanager, which is **not deployed** — alerts
are pull-only for now.

Rules are unit-tested behaviourally against synthetic series in
`monitoring/tests/gpu_rules_test.yml`:

```bash
~/bin/prometheus/promtool test rules monitoring/tests/gpu_rules_test.yml
```

## Service management

```bash
# Status of the whole stack
systemctl --user status prometheus.service node_exporter.service gpu-metrics.service

# Start / stop / restart
systemctl --user start prometheus.service
systemctl --user restart prometheus.service
systemctl --user stop prometheus.service

# Logs (follow)
journalctl --user -u prometheus.service -f
tail -f ~/.local/state/prometheus/logs/prometheus.log
tail -f ~/.local/state/node_exporter/logs/node_exporter.log
tail -f ~/.local/state/node_exporter/logs/gpu-metrics.log

# Reload after editing monitoring/prometheus.yml or a rule file:
# the lifecycle API is disabled, so restart the unit.
systemctl --user restart prometheus.service
```

## Troubleshooting

### Prometheus will not start

| Log message | Cause / fix |
|---|---|
| `error loading config ... no such file or directory` on a rule file | `rule_files` paths are resolved relative to `monitoring/prometheus.yml`, so they must be bare file names — see [Rule paths](#rule-paths-are-relative-to-the-config-file) |
| `cannot assign requested address` | A `--web.listen-address` does not exist on this host (tailnet down, or the node was re-registered) — see [Changing the boundary](#changing-the-boundary) |
| `permission denied` on the data directory | `~/.local/share/prometheus/data` must be owned by the user running the unit |
| `bind: address already in use` | Something else holds `:9090` — `ss -tlnp \| grep 9090` |

### A scrape target is DOWN

- `llama-proxy`: `curl http://localhost:8000/health`, then
  `systemctl --user status llama-proxy.service`.
- `node-exporter`: `curl http://localhost:9100/metrics`,
  `systemctl --user status node_exporter.service`,
  `tail ~/.local/state/node_exporter/logs/node_exporter.log`.
- A `rocm-exporter` target must never reappear: there is nothing to install and
  the job is commented out of `monitoring/prometheus.yml`.

### GPU series are missing or frozen

```bash
systemctl --user status gpu-metrics.service
ls -l --time-style=+%s ~/.local/share/node_exporter/textfile/gpu.prom   # mtime ≈ now
curl -s http://127.0.0.1:9100/metrics | grep '^gpu_'
```

If `gpu.prom` is missing, the sampler cannot read `gpu_busy_percent` (no AMD
GPU, or the driver is not loaded) and exits without writing, by design — see
[docs/systemd/gpu-metrics.service](../docs/systemd/gpu-metrics.service).

### `gpu_busy_percent` reads 100% while the machine is idle

Expected on this APU: the counter is pinned at 100% regardless of GPU state. Use
the power-derived series (`gpu:computing:bool`, `gpu:power_watts:avg5m`)
instead. No rule in this repository may select `gpu_busy_percent`
(LP-0MS9GA6QL007XTBP).

## Security notes

- **No authentication anywhere in this stack.** The exposure boundary is the
  tailnet — see [Exposure boundary](#exposure-boundary).
- **Prometheus can be stopped by any tailnet peer** (`POST /-/quit`). Lifecycle
  (`/-/reload`) and admin (delete-series) APIs are disabled; keep them disabled.
- **Exporters are loopback-only** (`127.0.0.1:9100`) and are scraped locally;
  publishing them adds no value and removes the network boundary.
- **Grafana, if deployed, needs attention**: it binds `:3000` on all interfaces
  by default and ships default credentials — bind it like Prometheus and change
  the password.
- **Never commit secrets** (Alertmanager webhook URLs, tokens) to
  `monitoring/`; alerting delivery is not implemented yet.

## Related work items

- LP-0MVO8CLCL00086NT — deployable, tailnet-exposed Prometheus with honest GPU
  utilisation (epic).
- LP-0MVO8DO7100309LH — `rule_files` paths are resolved relative to the config
  directory and blocked cold start.
- LP-0MVO8DOHP008FL48 — GPU utilisation must come from power; two alert groups
  were dead.
- LP-0MVO8DORX004P05G — this document.
- LP-0MS9GA6QL007XTBP — `gpu_busy_percent` pinned at 100% on this APU.
