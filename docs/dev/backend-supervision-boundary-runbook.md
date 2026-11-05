# Runbook: the llama-server supervision boundary (victim immunity)

Work item: **LP-0MW3PLK5M002YPOA** (second half of parent **AC2** —
"kills stop",
`LP-0MW2RI9AM007O1FO`). Supersedes the *name-based* hardening
branch of the kill hypothesis space: the decoy experiment in
[`docs/death-attribution-LP-0MW3K7WYV001XLL1.md`](../death-attribution-LP-0MW3K7WYV001XLL1.md)
§9 proved the sweep does **not** select `argv[0]`, and the class-A child
[`LP-0MW3PLEH8008O4I7`](../death-attribution-class-a-LP-0MW3PLEH8008O4I7.md)
identified the sender as this repo's own `start-proxy.sh --restart` (its port
fallback `fuser -k` selects by **port**). Name or argv hardening cannot stop
either vector, but a **boundary** can: the backend lives in its own
supervision unit/cgroup and process group, so a sweep of the proxy's group,
scope or session cannot reach it, and a sweep that *does* land is healed by
the unit's restart policy.

Tags: `SIGKILL`, `victim-immunity`, `systemd`, `cgroup`, `process-group`.

---

## 1. Verdict: boundary, not blocker

| Question | Answer |
|---|---|
| What is the threat? | A stray `SIGKILL`/`SIGTERM` sweep. The measured sender is the launcher's port fallback (`fuser -k`); the defence-in-depth concern is *any* neighbouring sweep (name, group, scope). |
| Why not name/argv hardening? | The decoy experiment: three `exec -a <backend path> sleep` decoys — none ever signalled. A name-pattern sweep is not the vector. |
| What is the boundary? | The router runs under its **own systemd unit** (`llama-server.service`, own cgroup `…/user@1000.service/llama-server.service`, own session + process group, `KillMode=control-group`, `Restart=always`), disjoint from the proxy's login-session scope (`…/session-N.scope`). |
| What if a kill still lands? | `Restart=always` + `RestartSec=2` brings the backend back within seconds instead of stranding it (compare: 2026-11-05 the proxy's own spawn failed and the router stayed down for hours). |
| Who must act? | Enabling the boundary is an operator action (`systemctl --user …`, no root). The **system-service variant needs root** (`sudo`), see §5. |

## 2. The boundary in one picture

```
user.slice/user-1000.slice/
├── user@1000.service/            ← the user manager (systemd --user)
│   ├── session-7.scope           ← the PROXY's login session (proxy, TTS)
│   ├── llama.slice/
│   │   └── llama-server.service  ← the backend: OWN cgroup, OWN PGID/SID
│   └── llama-proxy.service       ← the proxy under systemd (optional variant)
```

- **cgroup axis** — `systemctl --user kill` / a scope-targeted sweep of the
  proxy's session cannot reach `llama-server.service`'s cgroup.
- **process-group axis** — the backend (router + model workers) is one
  session/PGID, distinct from the proxy's, so `kill -TERM -<proxy-pgid>`
  cannot reach it.
- **restart axis** — even a direct hit is healed by `Restart=always`.

The proxy-side *signal* guard (disjoint-PGID kill refusal,
[`proxy/proxy/backend_isolation.py`](../../proxy/proxy/backend_isolation.py),
LP-0MW3K7YO1005C41X) remains the fail-closed backstop; this boundary is the
wider, durable layer.

## 3. Deploy the boundary (user service — no root)

The unit is documented in
[`docs/systemd/llama-server.service`](../systemd/llama-server.service)
(user-service block). The shipped unit sets `LLAMA_PARALLEL` to match the
proxy's `session_slot_pool_size` (2) — **keep these aligned**; a change in
either requires a `systemctl --user restart llama-server.service`.

```bash
# 1. Install the unit (edit the documented block as needed: binary, ports)
cp docs/systemd/llama-server.service ~/.config/systemd/user/llama-server.service
mkdir -p ~/.local/state/llama-server/logs

# 2. Reload, enable, start
systemctl --user daemon-reload
systemctl --user enable llama-server.service
systemctl --user start llama-server.service

# 3. Verify the backend is actually under the unit (not inline-spawned)
systemctl --user status llama-server.service --no-pager
```

**The proxy must then adopt the supervised backend** (it probes the port and
adopts a matching router instead of spawning a duplicate, config-hash-checked):
restart the proxy once after the unit is live:

```bash
cd ~/projects/llm/proxy
./scripts/start-proxy.sh --restart --reason "adopt unit-supervised backend (LP-0MW3PLK5M002YPOA)"
```

> **Why a restart is needed once:** the proxy refuses to *adopt* a backend
> whose merged-config hash differs from the one it recorded. After adoption,
> the proxy holds no process handle for the backend, so its stop path can no
> longer kill it directly — it stops the unit via the port/systemd contract
> instead. The unit's `Restart=always` guarantees even a stop is temporary.

## 4. Verify the boundary (evidence commands)

### 4.1 The backend's own cgroup (AC1)

```bash
# The router (the unit's main process)
systemctl --user show llama-server.service -p MainPID --value
cat /proc/$(systemctl --user show llama-server.service -p MainPID --value)/cgroup
# Expected: .../user@1000.service/llama.slice/llama-server.service
# Compare with the proxy's scope: .../user@1000.service/session-7.scope

# Process-group disjointness
ps -o pid,pgid,sid,comm -p $(systemctl --user show llama-server.service -p MainPID --value)
ps -o pid,pgid,sid,comm -p $(pgrep -f 'uvicorn proxy\.server' | head -1)

# The unit tree at a glance
systemd-cgls --user-unit llama-server.service
```

### 4.2 A sweep of the proxy's group leaves the backend alive (AC2/AC3)

Never point a live sweep at the real proxy while it serves traffic; the
command below is the **decoy demonstration** (a placeholder "neighbour"
process group that represents the proxy's session). The regression suite
runs the same scenario deterministically:
`proxy/tests/test_backend_boundary_isolation.py`.

```bash
# A decoy "neighbour" process (own group/session) — stands in for the proxy
bash -c 'sleep 300' &
NEIGHBOUR=$!

# The backend (the unit's router) — record its pid
BACKEND=$(systemctl --user show llama-server.service -p MainPID --value)

# Sweep the neighbour's process group: the backend must be untouched
kill -TERM -"$NEIGHBOUR" && echo "sweep executed"
sleep 1
kill -0 "$BACKEND" && echo "BACKEND ALIVE after the sweep" || echo "BACKEND GONE!"
pgrep -g "$BACKEND"   # the router's group is intact
ss -ltn | grep -q ':8080 ' && echo "port 8080 still listening"
```

Expected output: `sweep executed`, `BACKEND ALIVE after the sweep`, the
router's PGID listing, and `port 8080 still listening`.

### 4.2b Captured evidence (2026-11-05, live demo)

Captured on this host with a disposable supervised backend (a transient
``systemd-run`` unit — the same unit/cgroup mechanism the shipping unit
uses — and a disposable ``setsid`` proxy stand-in in the login session
scope). Nothing outside the demo was touched:

```
=== AC1: the backend's own supervision boundary (unit cgroup + PGID/SID) ===
backend pid=2241897
  cgroup: 0::/user.slice/user-1000.slice/user@1000.service/app.slice/boundary-demo-bd-ld-2241895.service
  pgid=2241897 sid=2241897
proxy stand-in pid=2241911
  cgroup: 0::/user.slice/user-1000.slice/session-7.scope
  pgid=2241911 sid=2241911

--- systemd-cgls (unit tree) ---
Unit boundary-demo-bd-ld-2241895.service (/user.slice/user-1000.slice/user@1000.service/app.slice/boundary-demo-bd-ld-2241895.service):
└─2241897 boundary-demo-backend-bd-ld-2241895 300

=== AC3: signal the PROXY's process group (kill -TERM -<proxy pgid>) ===
kill -TERM -2241911 sent
RESULT: backend ALIVE after proxy-group sweep (pid 2241897)

=== AC3b: name sweep of the neighbour (pkill -f <proxy marker>) ===
RESULT: backend ALIVE after neighbour name sweep (pid 2241897)
```

The demo backend's cgroup is under ``user@1000.service/...`` (its own unit),
the proxy stand-in's is the login session scope — a ``kill -TERM -<proxy
pgid>`` and a ``pkill -f`` sweep of the neighbour both leave the backend
running. The same boundary for the real router is one ``systemctl --user
start llama-server.service`` away (§3).

### 4.3 The `/health` verdict

The proxy surfaces the boundary read-only in `/health` (field
`supervision_boundary`):

```bash
curl -s localhost:8000/health | python3 -c 'import json,sys;print(json.load(sys.stdin)["supervision_boundary"])'
# state "isolated" when the backend lives outside the proxy's cgroup+group
# state "coupled"  when a backend is inline-spawned (immunity gap, visible)
```

`proxy/proxy/backend_boundary.py` implements the check (cgroup + PGID/SID
disjointness) and `proxy/tests/test_backend_boundary_isolation.py` covers it.

### 4.3b Adopted-backend liveness (LP-0MW3MOMD9007DC26)

An **adopted** backend — one the proxy found already running at startup and
installs the pid-0 sentinel for — holds no `Popen`, so the proxy cannot
`poll()` its death. It is **never** reported "running" from the sentinel
alone: `/health`'s `llama_server_running` and `/llama/local/status` both
require the port to answer (`backend_reachable`) for an adopted handle, and the
watchdog probes reachability every interval, declaring the death after
`llama_adopted_unreachable_probes` consecutive unreachable probes (default 3 ≈
15 s) and triggering the normal router self-heal. Symptom of a dead adopted
backend: `llama_server_running: false` / `backend_reachable: false` with a
`watchdog detected adopted llama-server death` line (or the pre-3.0 picture of
`running: true` + `backend_reachable: false` is the *pre-fix* bug signature).

## 5. Root/operator actions (system-service variant)

Only needed when the backend must survive logout or run system-wide:
**requires `sudo`** — copy the system block of the unit to
`/etc/systemd/system/llama-server.service` (`User=rgardler`, own
`Slice=llama.slice`), then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server.service
sudo systemctl start llama-server.service
sudo journalctl -u llama-server.service -f
```

No other root steps are required for the user-service deployment.

## 6. Known trade-offs

- **Slot-geometry changes.** `LLAMA_PARALLEL`/`--parallel` must match
  `session_slot_pool_size`; after changing either, restart the unit (the
  proxy exports the var at spawn time, which a unit-owned backend no longer
  receives).
- **Stop semantics.** `systemctl --user stop llama-server.service` is the
  *clean* way to stop the backend; `Restart=always` means a kill-style stop
  is only temporary by policy.
- **One backend, one supervisor.** Do not let the proxy spawn the router on
  a port the unit already holds (the proxy logs a bind-conflict and skips,
  but the clean model is adopt-only).
