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
| `llama-proxy.service` | n/a (Python/uvicorn) | `:8000` | scraped by Prometheus: proxy counters **and** the 10 s slot-occupancy sample (`llama_slots_*`, `llama_slot_busy`) | [docs/systemd/llama-proxy.service](../docs/systemd/llama-proxy.service) |
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

`llama-proxy`'s `/metrics` carries two kinds of series: the proxy's own
counters (requests, errors, backend retries) **and** a slot-occupancy sample
refreshed every 10 s (`server.llama_slot_metrics_interval_seconds`) from the
llama-server `/slots` path. Both are scraped every 15 s; together with the
`hwmon` power series they are what turns "the GPU drew 115 W" into our own
slot's work (see [GPU workload series](#gpu-workload-series-slot-occupancy)).
This unit (`gpu-metrics.service`) cannot see slot occupancy — the proxy owns
the only working `/slots` path.

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
> APU that counter **saturates at ~20 W** — it reads ~100 % once the GPU draws
> more than ~20 W and only falls back below that, so it reports idle-with-
> context stretches as busy (measured 2026-10-28 over 7 days: 94.5 % mean busy
> against 28.1 % of samples at compute power; LP-0MS9GA6QL007XTBP,
> quantified in LP-0MVSQQO910077LYL). `amd-smi`, `rocm-smi` and `amdtop` are
> not alternatives — see *Why not `amd-smi`, `rocm-smi` or `amdtop`?* in the
> query cookbook below.
> This power figure is **hardware** activity, never local model utilisation:
> it counts any GPU draw at compute power, not just the work the proxy served.
> The usage report's decision figure — the one gates and alerts quote — is the
> log-derived `local_busy`; see the
> [usage-analysis reference](../.pi/skills/proxy-usage-analysis/REFERENCE.md#which-busy-figure-is-authoritative-for-decisions).
> GPU memory pressure is measured on **GTT**, not the
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
  - "llama_memory_alerts.yaml"
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
# expected: SUCCESS: 6 rule files found
#           SUCCESS: monitoring/prometheus.yml is valid prometheus config file syntax
~/bin/prometheus/promtool test rules monitoring/tests/gpu_rules_test.yml
# expected: SUCCESS
systemctl --user restart prometheus.service
```

### llama-server memory series (llama_process_rss_bytes)

`llama_memory_alerts.yaml` alerts on `llama_process_rss_bytes`, published by
`proxy/proxy/backend_rss.py` and refreshed on **every `/metrics` scrape** (the
15 s Prometheus scrape alone keeps it fresh; no `/admin/metrics` poll is
needed). The value is resolved without requiring a host process handle:

- **podman-backed** (`scripts/podman_start_llama.sh`): the container init pid
  from `podman inspect -f '{{.State.Pid}}' <name>` and the summed RSS of its
  process tree, falling back to the container cgroup's `memory.current` (the
  number `podman stats` reports). Container name: `server.llama_container_name`,
  else `PODMAN_LC_CONTAINER`, else auto-detected from a `podman` start script.
- **host-direct**: the summed RSS of the llama-server process tree (router plus
  its per-model children — model memory lives in children, so a single-process
  read under-reports by ~99 %). The tracked `llama_process` pid is used when it
  is a real backend process; otherwise (adopted server with `pid = 0`, or a
  `podman exec` wrapper pid) pids are discovered by scanning `/proc` for
  `llama-server --port <llama_server_port>`.

Per-model attribution stays the documented equal split: the
`llama_model_rss_bytes{model=...}` labels sum to the process total. A runtime
lookup failure is fail-soft — the scrape still succeeds, the last known value
is kept, and the reason is logged once (not per scrape). Verify with:

```bash
curl -s localhost:8000/metrics | grep llama_process_rss_bytes   # non-zero
curl -s localhost:8000/admin/metrics | jq .process_rss_bytes    # same value
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
`gpu_vram_total_bytes`, `gpu_busy_percent` (scraped for continuity only — it
saturates at ~20 W on this APU and is not a utilisation signal), plus
`node_hwmon_power_average_watt{chip_name="amdgpu"}`.

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

The dashboard's GPU memory panel — "GPU memory: GTT (models) and VRAM
carve-out (bytes)" — charts three of the series § 3 lists as published:
`gpu_gtt_used_bytes` (the pool models map into) alongside
`gpu_vram_used_bytes` and `gpu_vram_total_bytes` (the ~1 GiB BIOS carve-out,
shown for context rather than as the model pool). Every expression is guarded
by `TestGrafanaDashboardVramPanel`, which derives the published set by running
the documented sampler, so a panel selecting an unpublished series fails the
suite instead of deploying as "No data". The earlier `rocm_vram_*` query is
gone: the AMD rocm-exporter route it assumed has no release artefacts (see
[rocm-exporter.service](../docs/systemd/rocm-exporter.service)) and Prometheus
has no such series to serve. Prometheus itself is unaffected.

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

# 5. Rules loaded: 5 groups, 11 rules
curl -s http://127.0.0.1:9090/api/v1/rules | jq -r '.data.groups[] | "\(.name) \(.rules|length)"'
# expected: gpu-compute.rules 5
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

### GPU workload fraction (power **and** a busy llama slot)

`gpu:computing:bool` answers "is the GPU at compute power?", which includes
work another process is doing. `gpu:computing_workload:bool` is the one shared
definition of "the GPU is doing **our** local model's work": 1 while power is
at or above the 90 W compute threshold **and** at least one llama-server slot is
processing, 0 otherwise. The CLI, the alert rules and ad-hoc queries all read
this series instead of re-deriving the condition.

```promql
# Share of the last 2h the GPU spent on OUR work (0..1)
avg_over_time(gpu:computing_workload:bool[2h])

# Our computing seconds in the last 24h
sum_over_time(gpu:computing_workload:bool[24h:15s]) * 15

# How much compute power we owned (power-only), for comparison
gpu:computing:bool
```

Three properties matter when querying it:

- **The join is label-agnostic.** The power series comes from the
  `node-exporter` job (`instance="localhost:9100"`) and `llama_slots_busy` from
  the `llama-proxy` job (`instance="localhost:8000"`), so the rule multiplies
  with `on()` (empty label set) rather than matching on any label. A naive
  `and on(instance)` join matches nothing and records no series at all — which
  looks like "unknown" forever, or like "never computing" on a dashboard.
- **Absence stays absent.** With `llama_slots_busy` missing (sampler or proxy
  down) or the power series missing (node_exporter down) the rule records **no
  series**, never 0. Absence is *unknown*; 0 is *idle*. Do not `or vector(0)`
  in a query unless you deliberately want unknown reported as idle.
- **The recorded labels are the power series' labels**
  (`instance="localhost:9100"`, `job="node-exporter"`); no llama-proxy labels
  leak into the combined series.

The behavioural cases (0/1 values, the 90 W boundary, and both missing-input
cases) live in `monitoring/tests/gpu_rules_test.yml`
(LP-0MVT1O4O1004ND6L).

#### GPU workload series: slot occupancy

These are the series the rule above joins on. They are published by the
**proxy** (`proxy/proxy/metrics.py`), not by `gpu-metrics.service`: the proxy
already owns the single working `/slots` path (child-port discovery, request
coalescing, failure backoff), so sampling from there costs no second poller.

| Series | Type | Meaning |
|---|---|---|
| `llama_slots_total` | gauge | slots reported by the last successful sample |
| `llama_slots_busy` | gauge | how many of those slots have `is_processing` true |
| `llama_slot_busy{slot="<id>"}` | gauge, 1/0 | per-slot state, so a single stuck slot is visible |
| `llama_slots_in_flight_tasks` | gauge | slots processing a task with an assigned `id_task` (falls back to the busy count on builds that publish no `id_task`) |
| `llama_slots_sample_timestamp_seconds` | gauge | heartbeat: the *age* of the last successful sample |

Cadence: the proxy samples every 10 s
(`server.llama_slot_metrics_interval_seconds`, default 10, deliberately below
the 15 s scrape); Prometheus observes it at 15 s.

```promql
# Occupancy right now: busy / total
llama_slots_busy
llama_slots_total

# Busy fraction over the last 2h (0..1). Number of busy slots, not a %.
avg_over_time(llama_slots_busy[2h]) / avg_over_time(llama_slots_total[2h])

# Which slots are stuck: 1 means that slot is processing
llama_slot_busy == 1

# Is the sampler still reporting? Age of the last successful sample, seconds.
time() - llama_slots_sample_timestamp_seconds
# expected: under ~15 s; a growing value means the sampler or the proxy died,
#           which is *unknown* occupancy, not zero occupancy

# Busy slot-minute count over a day
sum_over_time(llama_slots_busy[24h:15s]) * 15 / 60
```

Read a missing slot series as **unknown**, never as zero: the proxy publishes
nothing while its `/slots` sample is stale, and
`gpu:computing_workload:bool` then records no series either (see the three
properties under the workload rule above).

### Why not `amd-smi`, `rocm-smi` or `amdtop`?

The report and the alert rules measure GPU use from **power draw** because the
AMD tooling offers nothing better on this host (measured 2026-10-28,
`amd-smi` 26.2.2 / ROCm 7.2.4, single `gfx1151` APU device):

| Tool | What it gives here |
|---|---|
| `amd-smi` | `N/A` for engine usage, socket power, clocks, thermals and energy (`amd-smi metric`, `amd-smi monitor`); VRAM only (`0.3 / 1.0 GB`), which misdescribes an APU that serves from GTT |
| `rocm-smi` | power (the same `hwmon` `power1_average` node this stack scrapes) and `GPU use 100%` from `gpu_busy_percent` |
| `amdtop` | Linux “GPU usage” is literally `/sys/class/drm/card0/device/gpu_busy_percent` (`linux.go:151`, commit `5e500c3`); power/temp from the same `hwmon` files; VRAM (not GTT). Windows uses ADLX, a Windows-only driver library |

So `amdtop` and `rocm-smi` display **the same counter** this stack samples for
continuity only, and that counter **saturates at ~20 W**: it responds to load
within seconds but reads ~100 % for any draw above ~20 W. Measured over 7 days
of Prometheus history (904 paired 15 s samples, 2026-10-21 → 2026-10-28):

| GPU power band | samples | mean `gpu_busy_percent` | share ≥ 90 % |
|---|---|---|---|
| 0–20 W | 44 | 2.3 % | 2 % |
| 20–40 W | 414 | 98.9 % | 99 % |
| 40–60 W | 90 | 100.0 % | 100 % |
| 80–90 W | 96 | 100.0 % | 100 % |
| 90–100 W | 175 | 99.1 % | 98 % |
| 100–125 W | 78 | 95.7 % | 95 % |

Window summary: mean 94.5 % busy against 28.1 % of samples at compute power
(≥ 90 W); `r(busy, power) = 0.28`; false-busy (`busy ≥ 90 %` ∧ `power < 40 W`)
45.2 % of samples, longest continuous run 5.3 h; false-idle
(`busy < 90 %` ∧ `power ≥ 90 W`) 0.8 %, so it never *under*-reports — which is
why a live reading looks convincing.

Two structural points apply even if a tool did expose a usable number:
`amd-smi` and friends are **instantaneous** (no history, so a 24 h window would
need a new sampler), and where they work they read the **same** `hwmon` sensor
node_exporter already scrapes. The retained source is therefore
`hwmon` → Prometheus → `gpu:computing:bool` (see
[gpu_compute_rules.yaml](gpu_compute_rules.yaml)).

<!-- LP-0MVSQQO910077LYL -->

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

### Retention and resolution

**Configured retention is 90 days** (`--storage.tsdb.retention.time=90d`; the
`prometheus_tsdb_*` self-metrics are not scraped by this instance, so read it
from the flags API):

```bash
curl -s http://127.0.0.1:9090/api/v1/status/flags \
  | jq -r '.data["storage.tsdb.retention.time"]'
# expected: 90d
```

**90 d is a ceiling, not a claim about available data: history starts when the
TSDB was created.** On this host the store was created on **2026-10-25 at
19:22 UTC** — the first retained scrape is `2026-10-25T19:22:55Z`, and the
`hwmon` GPU power series starts slightly later, at
`2026-10-25T20:59:50Z` (the collector was added after the first boot). Anything
older than the TSDB creation time does not exist and no query can recover it.

```promql
# Oldest sample held by the TSDB, as an epoch timestamp.
# A subquery is required: an instant vector cannot see history, so
# min(timestamp(up)) on its own returns *now*, not the oldest sample.
min_over_time(timestamp(up)[90d:1h])

# How much history is actually retained, in days
(time() - min_over_time(timestamp(up)[90d:1h])) / 86400
```

The `1h` subquery step locates the first sample to within an hour; use
`[90d:1m]` when the exact second matters (slower, ~130 k evaluation points).

**Resolution floor is 15 s.** Three cadences are stacked, and the *coarsest*
one binds:

| Stage | Cadence |
|---|---|
| `gpu-metrics.service` sysfs sampler writes `gpu.prom` | 10 s |
| proxy slot-occupancy sample onto `/metrics` | 10 s |
| Prometheus scrapes both targets | **15 s** |

So the finest fidelity the store holds is one sample per 15 s per series; the
10 s writers are deliberately faster so no scrape misses a fresh value, not so
that a 10 s claim can be made about the retained data. Slot-occupancy figures
are additionally limited by the sampler: a task shorter than one sample
interval can fall between two samples and show as zero busy slots.

`scripts/gpu_utilisation.py` prints the resolution it actually used, never a
finer one — the default step is the 15 s scrape interval, `--step` may only
coarsen it, and a finer `--step` is rejected as a usage error:

```text
  resolution: step 15s (scrape floor 15s); 481 expected samples, 0 observed, 481 unknown
```

## Three "busy" figures — what each answers

The stack publishes three different measures that all look like "GPU usage".
They answer different questions, and none replaces another.

| Figure | Source | Answers | Does not answer |
|---|---|---|---|
| `gpu:computing:bool` (`avg_over_time(...)` = computing fraction) | `node_hwmon_power_average_watt` ≥ 90 W — hardware only | "was the GPU drawing compute power?" (any consumer: us, another process, a batch job) | whether the work was *ours* |
| `gpu:computing_workload:bool` (the CLI's `computing` share) | power ≥ 90 W **and** ≥ 1 busy llama slot | "did the GPU draw compute power while one of our slots was processing?" — hardware attributed to our slots | wall-clock slot occupancy: a single short task inside one 15 s scrape still counts as a computing instant |
| `local_busy.busy_pct` (the usage report's decision figure) | proxy log, `dispatch → finish` slot union — **proxy-inferred**, no hardware | "how much wall-clock time did our slots spend on tasks?" — the figure gates and alerts quote | how much GPU power that work drew, or any GPU work outside the proxy |

The log-derived and power-based figures end up on different scales and
legitimately diverge. Measured
2026-10-27 22:10 → 2026-10-28 22:10 the log-derived figure was **12.6 %**
against a power-based **39.3 %** — a ~27-point gap (LP-0MVSPWLLP008AQXZ). The
causes are structural: the log union counts queued slot time and misses work it
never logged, while the power figure counts *any* GPU draw at compute power,
including work the proxy never served, and (because a 15 s scrape is binarised
at 90 W) rounds a 2 s burst up to a whole sample.

Read them as answers to different questions:

- **Is local capacity contended?** → `local_busy.busy_pct`. This is the
  decision figure; see
  [which busy figure is authoritative](../.pi/skills/proxy-usage-analysis/REFERENCE.md#which-busy-figure-is-authoritative-for-decisions).
- **Did our slots actually keep the GPU computing?** →
  `gpu:computing_workload:bool` (or `scripts/gpu_utilisation.py`).
- **Is the GPU busy at all, whoever asked?** → `gpu:computing:bool`.

## GPU utilisation CLI (`scripts/gpu_utilisation.py`)

`scripts/gpu_utilisation.py` answers "how busy was the GPU between 17:00 and
19:00?" from the deployed Prometheus store — never from `proxy.log` — so the
figure survives log rotation and needs no log parsing. It reports the
`gpu:computing_workload:bool` share (our work), the power-only share (any
consumer) and an explicit `unknown` share, and it fails loudly rather than
reporting missing data as idle.

```bash
cd /path/to/llm
python3 scripts/gpu_utilisation.py --window 2h
```

Real run on this host (2026-10-31 22:45 UTC), shown verbatim including the
stderr line, while the proxy's slot sampler was not reporting (exit code `4`):

```text
gpu_utilisation: no observed samples in 2026-10-31T20:45:45Z .. 2026-10-31T22:45:45Z (unknown 100.0%): the sampling pipeline was not reporting, so the GPU was neither busy nor idle in any measurable sense. Check prometheus.service, node_exporter.service and the proxy's slot sampler.
GPU utilisation 2026-10-31T20:45:45Z .. 2026-10-31T22:45:45Z (2h)
  computing 0.0% / idle 0.0% / unknown 100.0%
  power-only computing 13.1% (gpu:computing:bool: any GPU consumer, not just ours)
  power mean 55 W / max 120 W (481 samples); peak n/a busy slots
  resolution: step 15s (scrape floor 15s); 481 expected samples, 0 observed, 481 unknown
```

That is the contract working, not a failure: without `llama_slots_busy` the
`computing`/`idle` classification is impossible, so every instant is `unknown`
and the exit code is non-zero. Once the sampler reports, the same command
prints the same five lines with populated `computing`/`idle` shares, a
`peak <n> busy slots` figure and a non-zero `observed` count. The numbers move
with the window and the machine — the **layout** is what is stable.

A fixed window, for a report or an incident write-up:

```bash
python3 scripts/gpu_utilisation.py \
  --since 2026-10-31T02:13:00Z --until 2026-10-31T04:13:00Z --json
```

| Flag | Meaning |
|---|---|
| `--window <duration>` | window ending now: `30m`, `2h`, `24h`, `7d` (default `2h`) |
| `--since <ISO8601> --until <ISO8601>` | explicit window; may not be combined with `--window` |
| `--step <duration>` | sample step; must be ≥ the 15 s scrape interval (default 15 s) |
| `--scrape-interval <duration>` | store resolution floor and slot-heartbeat freshness bound (default `15s`); change it only when the Prometheus scrape interval changes |
| `--json` | machine-readable output (stable `schema_version: 1`) |
| `--prometheus-url <url>` | override `http://127.0.0.1:9090` |

`--json` carries `window`, `resolution`, `coverage` (expected / observed /
computing / idle / unknown sample counts), `shares_pct`, `power_watts`,
`slots.peak_busy`, `rule_cross_check` (the CLI recomputes the rule and reports
mismatches) and `warnings`. Percentages are rounded to four decimal places and
the sample counts are exact, so nothing is lost:

```bash
python3 scripts/gpu_utilisation.py --window 2h --json | jq '.shares_pct, .resolution'
```

Exit codes: `0` success, `1` unexpected error, `2` invalid arguments,
`3` Prometheus unreachable or answering with an error, `4` no observed samples
in the window (all-unknown; the message says so and the JSON is still emitted).

`gpu_busy_percent` is never queried by the CLI (a regression test enforces
it): it is not a utilisation signal on this APU, as explained in
[Why not `amd-smi`, `rocm-smi` or `amdtop`?](#why-not-amd-smi-rocm-smi-or-amdtop).

## GPU power calibration (and re-measuring it)

The 90 W compute threshold is the one calibration constant shared by this
stack and the proxy: `gpu:computing:bool` is
`gpu:power_watts:amdgpu >= bool 90` in
[gpu_compute_rules.yaml](gpu_compute_rules.yaml), and the proxy's GPU-wedge
detector reads `server.llama_gpu_wedge_power_low_threshold_watts: 90` from
`proxy/config.yaml`. They are the same number in two places and **must be
changed together**, or the alerting layer and the proxy disagree about what
"computing" means.

Measured bands on this host (Strix Halo APU, Radeon 8060S iGPU, `gfx1151`),
over 7 days of 15 s samples (2026-10-25 21:00 → 2026-10-31 22:00 UTC):

| State | GPU package power | Evidence from that window |
|---|---|---|
| deep idle (display off, no GPU context) | ~4–15 W | min 4.1 W, p1 11.1 W |
| idle with display and models resident | **~33–54 W** | p5 33.0 W, p25 35.1 W, **p50 45.0 W** |
| computing (local generation) | **~90–117 W**, typically ≈115 W | p75 92.0 W, p95 108.1 W, p99 117.1 W |
| peak observed | 131.0 W | max over the window |

Over the same window: mean 61.0 W and 29.7 % of samples at or above 90 W. The
90 W threshold sits between the idle bulk (p50 45 W) and the computing bulk
(≥ p75 92 W, typically ≈115 W). That band is not empty — 3,738 of 35,063 samples (10.7 %) fell
between 55 and 89 W, which is what transitions, partial loads and short bursts
look like — but it is the thinnest part of the distribution, and 90 W is where
a sample is far more likely to be computing than idling. A machine whose
*idle* band reached 90 W would make the rule fire on an idle GPU, so re-measure
after a hardware, driver or kernel change before trusting the figure.

### Re-measuring

`power1_average` is the sensor this stack already uses
(`node_hwmon_power_average_watt`, restricted to `chip_name="amdgpu"` by the
recording rule), so measuring it directly gives the same number Prometheus
would:

1. **Idle.** Leave the GPU alone for a couple of minutes (display on, models
   loaded is the realistic case), then read the sensor a few times:

   ```bash
   for i in 1 2 3 4 5; do
     cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_average
     sleep 2
   done
   ```

   The file is in **microwatts** (the hwmon convention), so divide by
   `1000000`: `59081000` → **59.1 W**. The `hwmon*` path varies by boot — the
   glob above is deliberate; pin it (`ls -d .../hwmon*`) if you need one path.
2. **Computing.** Start a known-long local generation (a large prompt, or a
   request with a big `max_tokens`) and read the same file while it runs. Read
   it during *prefill* or a long decode, not after the stream ends — power
   falls back within a second or so:

   ```bash
   # in another terminal, while the generation is streaming
   watch -n1 'cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_average'
   ```
3. **Cross-check against the retained series** for the same minutes — this is
   the figure the rules actually evaluate, and it proves the pipeline sees the
   same sensor:

   ```bash
   curl -s --get http://127.0.0.1:9090/api/v1/query \
     --data-urlencode 'query=max_over_time(gpu:power_watts:amdgpu[5m])'
   ```
4. **Update both thresholds together** if the bands moved — the rule
   expression in [gpu_compute_rules.yaml](gpu_compute_rules.yaml)
   (`>= bool 90`) *and* `server.llama_gpu_wedge_power_low_threshold_watts` in
   `proxy/config.yaml` — then validate and apply:

   ```bash
   ~/bin/prometheus/promtool check rules monitoring/gpu_compute_rules.yaml
   ~/bin/prometheus/promtool test rules monitoring/tests/gpu_rules_test.yml
   systemctl --user restart prometheus.service   # lifecycle API is disabled
   ```

   If the new threshold changes the expected 0/1 boundaries in
   `monitoring/tests/gpu_rules_test.yml`, update the synthetic series in the
   same commit — the tests state the threshold explicitly, so a stale test is
   a loud failure rather than a silent one.

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

There are also four **recording** rules in `gpu_compute_rules.yaml`:
`gpu:power_watts:amdgpu`, `gpu:computing:bool`, `gpu:computing_workload:bool`
(power ≥ 90 W **and** ≥ 1 busy llama slot — the shared definition of our local
model's work) and `gpu:power_watts:avg5m`.

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
GPU, or the driver is not loaded) and exits without writing, by design — the
counter is only a liveness check here, never a utilisation signal; see
[docs/systemd/gpu-metrics.service](../docs/systemd/gpu-metrics.service).

### `gpu_busy_percent` reads 100% while the machine is idle

Expected on this APU: the counter latches to ~100 % as soon as the GPU draws
more than ~20 W and only falls back below that, so an idle GPU holding a
residual context still reads as busy. It is **not** a utilisation measure. Use
the power-derived series (`gpu:computing:bool`, `gpu:power_watts:avg5m`)
instead. No rule in this repository may select `gpu_busy_percent`
(LP-0MS9GA6QL007XTBP, quantified in LP-0MVSQQO910077LYL).

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
- LP-0MS9GA6QL007XTBP — `gpu_busy_percent` is not a utilisation signal on this
  APU.
- LP-0MVSQQO910077LYL — why the GPU-utilisation source stays power draw
  (`amd-smi` returns `N/A`; `amdtop`/`rocm-smi` surface the saturating
  `gpu_busy_percent`; 7-day quantification).
- LP-0MVNWUEY8008XM6W — slot-occupancy telemetry epic (this documentation
  change is its AC7/AC8 slice).
- LP-0MVT1O0VG00905EL — the proxy publishes the slot-occupancy series this
  document queries.
- LP-0MVT1O4O1004ND6L — the combined `gpu:computing_workload:bool` rule and
  its behavioural tests.
- LP-0MVT1O52G000LMC6 — `scripts/gpu_utilisation.py`, the CLI documented here.
- LP-0MVSPWLLP008AQXZ — the log-derived `local_busy` is the decision figure;
  the ~27-point divergence from the power figure.
