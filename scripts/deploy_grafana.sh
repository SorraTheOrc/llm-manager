#!/usr/bin/env bash
# scripts/deploy_grafana.sh — idempotent, verifiable Grafana deployment.
#
# One command that installs Grafana 11.6.0 into ~/bin/grafana, provisions the
# Prometheus datasource and the `LLama Server Memory` dashboard, installs the
# systemd user unit extracted from docs/systemd/grafana-server.service, starts
# it loopback-bound (127.0.0.1:3000), publishes it on the tailnet via
# `tailscale serve` and proves the dashboard loaded from Grafana's own APIs.
#
# Usage:
#   scripts/deploy_grafana.sh [--check] [--allow-absent] [--quiet] [--version X.Y.Z]
#
#     (default)                      install + provision + enable + start + verify
#     --check                        verify only; never changes anything
#     --allow-absent                 with --check, exit 0 when Grafana is not deployed
#     --quiet                        report failures only
#     --version                      override the pinned Grafana version (default 11.6.0)
#     --extract-unit                 print the unit extracted from docs/systemd/grafana-server.service
#     --render-provider-config DIR   write the dashboard provider config for DIR and print it
#
# `--extract-unit` and `--render-provider-config` are pure helpers so tests can
# exercise the two rendering steps on an undeployed host without installing
# anything: no network, no writes outside the given path.
#
# Exit codes: 0 success; 1 deploy/verify failure; 2 usage error.
#
# The admin password lives in ~/.config/grafana/admin.pass (0600, directory
# 0700), is generated once and is never printed, logged, embedded in the unit
# or passed on a command line — Grafana reads it via
# GF_SECURITY_ADMIN_PASSWORD__FILE and API probes authenticate with a throwaway
# curl config file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GRAFANA_VERSION="11.6.0"
GRAFANA_PORT="3000"
GRAFANA_HOME="$HOME/bin/grafana"
SECRET_FILE="$HOME/.config/grafana/admin.pass"
LAN_ADDR="192.168.0.199"
PROM_URL="http://127.0.0.1:9090"

MODE="deploy"
QUIET=0
ALLOW_ABSENT=0
RENDER_DIR=""
AUTH_CFG=""
FAIL_DETAIL=""
DATASOURCE_UID=""

usage() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

say() { [ "$QUIET" = "1" ] || echo "$@"; }
warn() { echo "deploy_grafana: WARNING: $*" >&2; }
die() { echo "deploy_grafana: $*" >&2; exit 1; }

trap 'rm -f "${AUTH_CFG:-}"' EXIT

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    --allow-absent)
      ALLOW_ABSENT=1
      shift
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --version)
      GRAFANA_VERSION="${2:?--version needs a value}"
      shift 2
      ;;
    --extract-unit)
      MODE="extract-unit"
      shift
      ;;
    --render-provider-config)
      MODE="render-provider-config"
      RENDER_DIR="${2:?--render-provider-config needs a directory}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "deploy_grafana: unknown argument '$1'" >&2
      echo "  Try: $(basename "$0") --help" >&2
      exit 2
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Pure rendering helpers (usable on an undeployed host, no network)
# ---------------------------------------------------------------------------

# extract_unit: print the fenced ini block of docs/systemd/grafana-server.service
# — the same `unit_from_doc` awk extraction documented in monitoring/README.md.
extract_unit() {
  awk '/^```ini$/{f=1;next} /^```$/{f=0} f' \
    "$REPO_ROOT/docs/systemd/grafana-server.service"
}

# render_provider_config DIR: write the dashboard provider config with
# options.path replaced by the absolute dashboards directory DIR, into
# DIR/dashboards.yaml, and print the rendered config.
render_provider_config() {
  local dest_dir="$1" dashboards_dir
  mkdir -p "$dest_dir"
  dashboards_dir="$(cd "$dest_dir" && pwd)"
  sed "s|^\([[:space:]]*path:\).*|\1 $dashboards_dir|" \
    "$REPO_ROOT/monitoring/grafana/dashboards/dashboards.yaml" \
    > "$dashboards_dir/dashboards.yaml"
  cat "$dashboards_dir/dashboards.yaml"
}

# _version_matches: 0 when `grafana-server -v` output reports the pinned
# version. The binary prints e.g. "Version 11.6.0 (commit: …)" (with a
# deprecation warning line before it on 11.x) — no bare `v` prefix to glob on.
_version_matches() { # $1 = grafana-server -v output
  case "$1" in
    *"Version $GRAFANA_VERSION"*) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Preflight, install
# ---------------------------------------------------------------------------

preflight() {
  if ! curl -fsS --max-time 5 "$PROM_URL/-/healthy" >/dev/null 2>&1; then
    die "Prometheus ($PROM_URL) is not answering /-/healthy — refusing to deploy a dashboard with a dead datasource. Start it with: systemctl --user start prometheus.service"
  fi
  mkdir -p "$HOME/bin"
  # The unit's StandardOutput/StandardError are append targets under
  # ~/.local/state/grafana/logs — systemd cannot open an append: path whose
  # parent dir does not exist, so the state dirs must exist before the unit
  # starts (observed as status=209/STDOUT on a fresh host).
  mkdir -p "$HOME/.local/share/grafana/data" "$HOME/.local/state/grafana/logs"
  if ! command -v systemctl >/dev/null 2>&1; then
    die "systemctl not found — a systemd user session is required to install the grafana-server unit."
  fi
  if ! systemctl --user show-environment >/dev/null 2>&1; then
    die "the systemd user instance is unreachable (systemctl --user) — no user session?"
  fi
}

install_grafana() {
  local arch="amd64" tmp tarball expected actual extracted v url
  url="https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${arch}.tar.gz"
  tmp="$(mktemp -d)"
  tarball="$tmp/grafana.tar.gz"

  say "deploy_grafana: downloading $url"
  curl -fsSL --retry 3 --max-time 600 -o "$tarball" "$url" \
    || { rm -rf "$tmp"; die "failed to download $url"; }
  curl -fsSL --retry 3 --max-time 60 -o "$tarball.sha256" "$url.sha256" \
    || { rm -rf "$tmp"; die "failed to download $url.sha256"; }

  expected="$(awk '{print $1}' "$tarball.sha256")"
  actual="$(sha256sum "$tarball" | awk '{print $1}')"
  if [ -z "$expected" ]; then
    rm -rf "$tmp"
    die "the published checksum file for $GRAFANA_VERSION is empty — refusing to install."
  fi
  if [ "$expected" != "$actual" ]; then
    rm -rf "$tmp"
    die "Grafana checksum mismatch (expected $expected, downloaded $actual) — aborting; the existing installation at $GRAFANA_HOME was not touched."
  fi
  say "deploy_grafana: sha256 verified ($actual)"

  tar -xzf "$tarball" -C "$tmp"
  # The archive's top-level directory is grafana-v11.6.0 (Grafana's release
  # tag naming); accept either the v-prefixed or bare form.
  extracted="$(find "$tmp" -maxdepth 1 -type d -name 'grafana-*' | head -1)"
  if [ -z "$extracted" ] || [ ! -d "$extracted" ]; then
    rm -rf "$tmp"
    die "the tarball did not contain a grafana-* directory — refusing to install."
  fi
  mkdir -p "$GRAFANA_HOME"
  cp -a "$extracted/." "$GRAFANA_HOME/"
  rm -rf "$tmp"

  v="$("$GRAFANA_HOME/bin/grafana-server" -v 2>&1 || true)"
  if ! _version_matches "$v"; then
    die "installed grafana-server reports '$v', expected Version $GRAFANA_VERSION"
  fi
  say "deploy_grafana: installed grafana-server $GRAFANA_VERSION at $GRAFANA_HOME/bin/grafana-server"
}

generate_secret() {
  if [ -f "$SECRET_FILE" ]; then
    say "deploy_grafana: admin password file $SECRET_FILE already present; keeping it"
    return 0
  fi
  local pw
  install -d -m 0700 "$(dirname "$SECRET_FILE")"
  pw="$(head -c 32 /dev/urandom | base64 | tr -d '\n')"
  ( umask 077; printf '%s' "$pw" > "$SECRET_FILE" )
  chmod 600 "$SECRET_FILE"
  say "deploy_grafana: generated admin password file $SECRET_FILE (0600); the value is written to the file only"
}

provision() {
  local ds_src="$REPO_ROOT/monitoring/grafana/datasources/datasources.yaml"
  local ds_out="$GRAFANA_HOME/conf/provisioning/datasources/datasources.yaml"
  local dash_dir="$GRAFANA_HOME/conf/provisioning/dashboards"
  mkdir -p "$(dirname "$ds_out")" "$dash_dir"
  cp "$ds_src" "$ds_out"
  render_provider_config "$dash_dir" >/dev/null
  cp "$REPO_ROOT/monitoring/grafana_llama_memory_dashboard.json" "$dash_dir/"
  say "deploy_grafana: provisioned datasource config, dashboard provider and dashboard JSON"
}

install_unit() {
  local unit_dir="$HOME/.config/systemd/user" enabled=1 hint
  mkdir -p "$unit_dir"
  extract_unit > "$unit_dir/grafana-server.service"
  systemctl --user daemon-reload
  if ! systemctl --user enable --now grafana-server.service >/dev/null 2>&1; then
    enabled=0
  fi
  sleep 2
  if ! systemctl --user is-active --quiet grafana-server.service; then
    systemctl --user stop grafana-server.service >/dev/null 2>&1 || true
    systemctl --user disable grafana-server.service >/dev/null 2>&1 || true
    if [ "$enabled" = 0 ]; then
      hint="enable --now failed (is the unit valid? systemd-analyze verify $unit_dir/grafana-server.service)"
    else
      hint="the unit did not stay active after enable --now"
    fi
    die "grafana-server.service — $hint; it was stopped and disabled. Diagnose with: journalctl --user -u grafana-server.service -n 50"
  fi
  say "deploy_grafana: unit installed and active (grafana-server.service)"
}

expose() {
  if command -v tailscale >/dev/null 2>&1 && tailscale status >/dev/null 2>&1; then
    if tailscale serve --bg --tcp="$GRAFANA_PORT" "http://127.0.0.1:$GRAFANA_PORT" >/dev/null 2>&1; then
      local tip
      tip="$(tailscale ip -4 2>/dev/null | head -1 || true)"
      say "deploy_grafana: published on the tailnet via tailscale serve --tcp=$GRAFANA_PORT (http://${tip:-<tailnet-ip>}:$GRAFANA_PORT)"
    else
      warn "tailscale serve failed; Grafana stays loopback-only (http://127.0.0.1:$GRAFANA_PORT)"
    fi
  else
    warn "tailscale is not up; Grafana stays loopback-only (http://127.0.0.1:$GRAFANA_PORT)"
  fi
}

# ---------------------------------------------------------------------------
# Verification probes — shared by the deploy's verify phase and --check
# ---------------------------------------------------------------------------

# _auth_cfg: lazily create a throwaway 0600 curl config carrying the stored
# admin credential, so the secret never appears on any command line.
_auth_cfg() {
  if [ -z "$AUTH_CFG" ]; then
    [ -r "$SECRET_FILE" ] || { echo "deploy_grafana: admin secret file $SECRET_FILE is not readable" >&2; return 1; }
    AUTH_CFG="$(mktemp)"
    chmod 600 "$AUTH_CFG"
    {
      printf 'user = "admin:'
      cat "$SECRET_FILE"
      printf '"\n'
    } > "$AUTH_CFG"
  fi
  printf '%s\n' "$AUTH_CFG"
}

_api_get() { # $1 path; authenticated GET against loopback; prints body; fails on error
  local path="$1"
  curl -fsS --max-time 15 -K "$(_auth_cfg)" "http://127.0.0.1:$GRAFANA_PORT$path" 2>/dev/null
}

probe_binary_version() {
  [ -x "$GRAFANA_HOME/bin/grafana-server" ] \
    || { FAIL_DETAIL="no binary at $GRAFANA_HOME/bin/grafana-server"; return 1; }
  local v
  v="$("$GRAFANA_HOME/bin/grafana-server" -v 2>&1 || true)"
  if _version_matches "$v"; then return 0; fi
  FAIL_DETAIL="installed version does not match pinned $GRAFANA_VERSION (reports: $(printf '%s' "$v" | grep -o 'Version [0-9.]*' | head -1 || echo unknown))"
  return 1
}

probe_unit_enabled() {
  systemctl --user is-enabled --quiet grafana-server.service \
    || { FAIL_DETAIL="grafana-server.service is not enabled"; return 1; }
}

probe_unit_active() {
  systemctl --user is-active --quiet grafana-server.service \
    || { FAIL_DETAIL="grafana-server.service is not active (systemctl --user status grafana-server.service)"; return 1; }
}

probe_health() {
  local body db
  body="$(curl -fsS --max-time 10 "http://127.0.0.1:$GRAFANA_PORT/api/health" 2>/dev/null)" \
    || { FAIL_DETAIL="GET /api/health failed — Grafana not answering on 127.0.0.1:$GRAFANA_PORT"; return 1; }
  db="$(printf '%s' "$body" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("database",""))' 2>/dev/null)"
  [ "$db" = "ok" ] \
    || { FAIL_DETAIL="GET /api/health database is '${db:-<unset>}', expected 'ok'"; return 1; }
}

probe_datasource() {
  local body rc
  body="$(_api_get "/api/datasources")" \
    || { FAIL_DETAIL="authenticated GET /api/datasources failed (basic auth with $SECRET_FILE)"; return 1; }
  DATASOURCE_UID="$(printf '%s' "$body" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(2)
for x in d:
    if x.get("name") == "Prometheus":
        print(x.get("uid", "")); break
' 2>/dev/null)"
  rc=$?
  [ "$rc" -ne 2 ] || { FAIL_DETAIL="could not parse /api/datasources response"; return 1; }
  case "$DATASOURCE_UID" in
    "") FAIL_DETAIL="Prometheus datasource not present in /api/datasources"; return 1 ;;
  esac
}

probe_dashboard() {
  local body
  body="$(_api_get "/api/search?type=dash-db")" \
    || { FAIL_DETAIL="authenticated GET /api/search?type=dash-db failed"; return 1; }
  printf '%s' "$body" | python3 -c '
import sys, json
d = json.load(sys.stdin)
titles = [str(x.get("title", "")) for x in d]
assert any("LLama Server Memory" in t for t in titles), titles
' 2>/dev/null || { FAIL_DETAIL="LLama Server Memory dashboard not returned by /api/search?type=dash-db"; return 1; }
}

probe_datasource_query() {
  [ -n "$DATASOURCE_UID" ] || { FAIL_DETAIL="no Grafana datasource uid"; return 1; }
  local body
  body="$(curl -fsS --max-time 15 -K "$(_auth_cfg)" --get --data-urlencode 'query=up' \
    "http://127.0.0.1:$GRAFANA_PORT/api/datasources/uid/$DATASOURCE_UID/resources/api/v1/query" 2>/dev/null)" \
    || { FAIL_DETAIL="datasource query through Grafana failed"; return 1; }
  printf '%s' "$body" | python3 -c '
import sys, json
d = json.load(sys.stdin)
res = (d.get("data") or {}).get("result") or []
assert len(res) > 0, "no result series"
' 2>/dev/null || { FAIL_DETAIL="datasource query returned no result series"; return 1; }
}

probe_secret_file() {
  [ -f "$SECRET_FILE" ] || { FAIL_DETAIL="admin password file missing at $SECRET_FILE"; return 1; }
  local mode dir_mode body
  mode="$(stat -c '%a' "$SECRET_FILE" 2>/dev/null || echo bad)"
  dir_mode="$(stat -c '%a' "$(dirname "$SECRET_FILE")" 2>/dev/null || echo bad)"
  [ "$mode" = "600" ] || { FAIL_DETAIL="admin password file mode is $mode, expected 600"; return 1; }
  [ "$dir_mode" = "700" ] || { FAIL_DETAIL="admin password directory mode is $dir_mode, expected 700"; return 1; }
  body="$(_api_get "/api/org")" \
    || { FAIL_DETAIL="basic auth with the stored credential ($SECRET_FILE) failed"; return 1; }
  printf '%s' "$body" | python3 -c 'import sys,json; json.load(sys.stdin)' 2>/dev/null \
    || { FAIL_DETAIL="authenticated response not JSON"; return 1; }
}

probe_listener() {
  # ss output is passed via LISTENER_SS (environment), not stdin, so the
  # python -c program and the parsed data do not compete for fd 0.
  local ss_out detail
  ss_out="$(ss -tln 2>/dev/null || true)"
  detail="$(LISTENER_SS="$ss_out" python3 -c '
import os, sys
port = sys.argv[1]
loopback = False
bad = []
for line in os.environ.get("LISTENER_SS", "").splitlines():
    if f":{port}" not in line:
        continue
    parts = line.split()
    if len(parts) < 5:
        continue
    local = parts[3]
    if local.startswith("127.0.0.1:"):
        loopback = True
    elif local.startswith(("*", "0.0.0.0:", "[::]")):
        bad.append("wildcard " + local)
    elif local.startswith("192.168."):
        bad.append("LAN " + local)
    elif local.startswith("100.") or local.startswith("[fd7a:"):
        pass  # tailscaled tailnet listener — the intended boundary
    else:
        bad.append(local)
if not loopback:
    bad.append("no 127.0.0.1 listener")
print("; ".join(bad) if bad else "ok")
sys.exit(1 if bad else 0)
' "$GRAFANA_PORT")" || { FAIL_DETAIL="listener probe: ${detail:-unparseable ss output}"; return 1; }
}

probe_lan_refused() {
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 "http://$LAN_ADDR:$GRAFANA_PORT/" 2>/dev/null || true)"
  [ "$code" = "000" ] \
    || { FAIL_DETAIL="LAN address $LAN_ADDR:$GRAFANA_PORT answered HTTP $code (expected connection refused)"; return 1; }
}

PROBES=(binary_version unit_enabled unit_active health datasource dashboard datasource_query secret_file listener lan_refused)

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

deploy_main() {
  local probe
  preflight
  if [ -x "$GRAFANA_HOME/bin/grafana-server" ]; then
    local v
    v="$("$GRAFANA_HOME/bin/grafana-server" -v 2>&1 || true)"
    if _version_matches "$v"; then
      say "deploy_grafana: grafana-server already installed; skipping download"
    else
      say "deploy_grafana: installed grafana-server does not match pinned $GRAFANA_VERSION; reinstalling"
      install_grafana
    fi
  else
    install_grafana
  fi
  generate_secret
  provision
  install_unit
  expose

  say "deploy_grafana: waiting for Grafana to answer /api/health"
  local i
  for i in $(seq 1 30); do
    curl -fsS --max-time 3 "http://127.0.0.1:$GRAFANA_PORT/api/health" >/dev/null 2>&1 && break
    sleep 2
    [ "$i" -lt 30 ] || {
      systemctl --user stop grafana-server.service >/dev/null 2>&1 || true
      systemctl --user disable grafana-server.service >/dev/null 2>&1 || true
      die "Grafana never became healthy within 60 s — the unit was stopped and disabled. Diagnose with: journalctl --user -u grafana-server.service -n 50"
    }
  done

  for probe in "${PROBES[@]}"; do
    if ! "probe_$probe"; then
      die "VERIFY FAILED — $probe: ${FAIL_DETAIL:-unknown}. Remediation: journalctl --user -u grafana-server.service -n 50, then re-run scripts/deploy_grafana.sh --check"
    fi
    say "  [ ok ] $probe"
  done
  say "deploy_grafana: deployed and verified — open the dashboard via the tailnet (http://<tailnet-ip>:$GRAFANA_PORT) or locally (http://127.0.0.1:$GRAFANA_PORT)"
}

check_main() {
  local probe failures=()
  if [ ! -x "$GRAFANA_HOME/bin/grafana-server" ]; then
    echo "deploy_grafana: not deployed — no grafana-server at $GRAFANA_HOME/bin/grafana-server (run scripts/deploy_grafana.sh to install)"
    [ "$ALLOW_ABSENT" = "1" ] && exit 0 || exit 1
  fi
  for probe in "${PROBES[@]}"; do
    if ! "probe_$probe"; then
      failures+=("$probe: ${FAIL_DETAIL:-unknown failure}")
    fi
  done
  if [ "${#failures[@]}" -eq 0 ]; then
    say "deploy_grafana: --check OK — installed, enabled, active, healthy, provisioned, loopback-only, LAN refused"
    return 0
  fi
  echo "deploy_grafana: --check FAILED — ${#failures[@]} problem(s):" >&2
  local f
  for f in "${failures[@]}"; do
    echo "  - $f" >&2
  done
  return 1
}

case "$MODE" in
  check) check_main ;;
  extract-unit) extract_unit ;;
  render-provider-config) render_provider_config "$RENDER_DIR" ;;
  *) deploy_main ;;
esac
