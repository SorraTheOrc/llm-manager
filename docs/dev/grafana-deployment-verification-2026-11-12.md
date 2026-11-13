# Grafana deployment verification — 2026-11-12

End-to-end record of deploying Grafana on this host with
`scripts/deploy_grafana.sh` (LP-0MWCQKN3C0034R6Z). Every result below was
observed on the host, not inferred. **No secret value appears in this
document**: only the credential *path* is recorded.

## Preconditions

| Fact | Observed value |
| --- | --- |
| Grafana installed | `~/bin/grafana` absent before the run; installed by the script |
| Prometheus | answering `127.0.0.1:9090/-/healthy` (HTTP 200) |
| Tailscale | `1.104.6`, self `sorra-agents-1`, tailnet IPv4 `100.79.231.101` |
| `tailscale serve` | available, but **refused without privileges** (see *Tailnet exposure*) |

## Deployment (first run)

```
$ scripts/deploy_grafana.sh
deploy_grafana: downloading https://dl.grafana.com/oss/release/grafana-11.6.0.linux-amd64.tar.gz
deploy_grafana: checksum verified (043fa7b8…35169f)
deploy_grafana: installed grafana-server 11.6.0 at ~/bin/grafana/bin/grafana-server
deploy_grafana: generated admin password file ~/.config/grafana/admin.pass (0600)
deploy_grafana: provisioned datasource config, dashboard provider and dashboard JSON
deploy_grafana: unit installed and active (grafana-server.service)
deploy_grafana: WARNING: tailscale serve failed; Grafana stays loopback-only
  [ ok ] binary_version
  [ ok ] unit_enabled
  [ ok ] unit_active
  [ ok ] health
  [ ok ] datasource
  [ ok ] dashboard
  [ ok ] datasource_query
  [ ok ] secret_file
  [ ok ] listener
  [ ok ] lan_refused
```

## Idempotence (second run, AC 2)

A second `scripts/deploy_grafana.sh` exited `0` and changed nothing:

```
deploy_grafana: grafana-server already installed; skipping download
deploy_grafana: admin password file …admin.pass already present; keeping it
… all ten probes [ ok ] …
```

Evidence that state was untouched: `stat` of `~/.config/grafana/admin.pass`
(mtime + size) and `systemctl --user show grafana-server -p
ActiveEnterTimestamp` were byte-identical before and after the second run.

## Verification probes

| Probe | Observed |
| --- | --- |
| `--check` exit code | `0` — `--check OK — installed, enabled, active, healthy, provisioned, loopback-only, LAN refused` |
| `GET /api/health` | `{"database": "ok", "version": "11.6.0"}` |
| `GET /api/datasources` | `Prometheus` → `http://localhost:9090` (type `prometheus`) |
| `GET /api/search?type=dash-db` | `LLama Server Memory` (uid `bg4396qjx7x1cf`) |
| GPU memory panel | `gpu_gtt_used_bytes` returns 1 series (`gpu="card0"`, `job="node-exporter"`) |
| Proxy panels | `llama_process_rss_bytes` 1 series, `llama_models_loaded` 1 series |
| VRAM panels | `gpu_vram_used_bytes` 1 series, `gpu_vram_total_bytes` 1 series |
| 5xx panel | `rate(proxy_http_errors_total{…status="5xx"}[5m])` — empty because no 5xx errors occurred (correct) |
| Listener (AC 5) | `ss -tln` → `127.0.0.1:3000` only; no `*:3000` |
| LAN refused (AC 5) | `http://192.168.0.199:3000` → connection refused (`000`) |
| Secret hygiene (AC 5) | `~/.config/grafana/admin.pass` mode `0600`, its dir `0700`; value never printed, never on a command line (probes use a temporary `0600` curl config) |

## Tailnet exposure (environment-gated)

`tailscale serve --bg --tcp=3000` is refused on this host:

```
sending serve config: Access denied: serve config denied
Use 'sudo tailscale set --operator=rgardler'.
```

`tailscaled` accepts serve configuration only from root or the configured
operator, and this account has no passwordless sudo. The script therefore takes
its documented degradation path (parent epic AC 4): a warning, and Grafana stays
loopback-only. The dashboard is reachable at `http://127.0.0.1:3000` but not yet
at `http://100.79.231.101:3000`.

The privileged step is tracked separately as **LP-0MWEHZB9P008WJUR** — it needs
one operator command (`sudo tailscale set --operator=rgardler`) and then the
script publishes on the tailnet with no change.

## Remaining step (operator)

```bash
sudo tailscale set --operator=rgardler
tailscale serve --bg --tcp=3000 http://127.0.0.1:3000
curl -s -o /dev/null -w '%{http_code}\n' http://100.79.231.101:3000/api/health   # expect 200
```

## Test suite

The full project suite passes via `/skill:test` with no regressions (see the
work-item comment for the run summary).
