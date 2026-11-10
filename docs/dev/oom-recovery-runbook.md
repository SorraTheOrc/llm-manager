# Operator runbook: recovering from an OOM kill of the proxy

**Scope.** The LLM proxy (or its llama-server backend) has been SIGKILLed by the
Linux OOM killer, or the host is close to it. This is the operational companion
to work item LP-0MVZ810NG008A3EW — the 2026-11-02T12:03:08Z **global OOM** that
took the proxy and its backend down together for ~10.5 minutes.

Related: [monitoring/README.md](../../monitoring/README.md) (alert rules),
[docs/systemd/llama-proxy.service](../systemd/llama-proxy.service) and
[docs/systemd/llama-server.service](../systemd/llama-server.service) (units),
[proxy/README.md](../../proxy/README.md) (launcher and restart behaviour).

## 1. Recognise it

Symptoms, in the order they usually appear:

| Signal | Where |
|---|---|
| `connection refused` on `:8000` for every consumer | clients, agents, herdr |
| `curl -sf localhost:8000/health` fails while `:8080` still answers | proxy down, router alive |
| a `proxy.log` with **no** `Shutting down` line — the last line is just before the kill | `proxy/logs/proxy.log*` |
| `systemctl --user status llama-proxy.service` shows `code=killed, status=9/KILL` | systemd |
| `Recovery: previous proxy log write … Ns before this start` at startup | `proxy.log` after a restart |
| an OOM kill line | `journalctl -k` / `/var/log/kern.log` |

Confirm the OOM (the 2026-11-02 event looked exactly like this):

```bash
# The kill, the constraint and the victim list
journalctl -k --since "-30 min" | grep -E 'oom-kill|Out of memory|Killed process'
# or: grep -E 'oom-kill|Killed process' /var/log/kern.log

# Was it a GLOBAL oom (proxy at fault) or a memory-cgroup oom (one unit)?
#   constraint=CONSTRAINT_NONE ... global_oom   -> host-wide, see §2
#   constraint=CONSTRAINT_MEMCG                 -> a unit hit its MemoryMax
```

The kernel's memory dump in the same event is the only record of the pressure:

```
Node 0 Normal free:245576kB min:238872kB ... all_unreclaimable? yes
gpu_active:101385112kB  gpu_reclaim:0kB
active_anon:16680992kB inactive_anon:8390284kB
```

`gpu_active` is **GPU-backed host memory** (amdgpu GTT). It has no procfs/sysfs
interface, so it appears in this dump only. The driver's own accounting of the
same pool is readable at any time:

```bash
cat /sys/class/drm/card0/device/mem_info_gtt_used
cat /sys/class/drm/card0/device/mem_info_gtt_total
grep -E 'MemTotal|MemAvailable' /proc/meminfo
```

On this APU `mem_info_gtt_total` (128 GiB) is **larger than physical RAM**
(124 GiB), which is why GTT must always be compared with `MemTotal`, never with
its own pool.

## 2. Recover now (target: < 1 minute with the units installed)

```bash
# 1. Is the proxy supervised? If yes it is probably already back.
systemctl --user status llama-proxy.service

# 2. If it is not supervised (launched by hand with start-proxy.sh), restart it:
cd ~/projects/llm/proxy
./scripts/start-proxy.sh --restart --reason "recovering from OOM kill"

# 3. Verify (the launcher detaches; poll health, do not trust its exit code)
curl -sf http://127.0.0.1:8000/health && echo OK
curl -s  http://127.0.0.1:8080/v1/models | head -c 400

# 4. Confirm the OOM ranking took effect (negative on the proxy, positive on the
#    backend). This is what prevents a repeat of the double kill.
cat /proc/$(pgrep -f 'uvicorn proxy\.server' | head -1)/oom_score_adj
for p in $(pgrep -f llama-server); do cat /proc/$p/oom_score_adj; done
```

Expected: the proxy's score is `-900` when started from the **system** unit
(root) or `0` when started unprivileged, and every llama-server process scores
`800`. A `0` proxy score means the negative value was refused for lack of
`CAP_SYS_RESOURCE` — the backend still ranks above the proxy, which is what
matters most; install the system unit to also protect the front door.

### Shed the pressure before reloading a model

A crash-restart loop is worse than a short outage: if the host is still full,
the restarting proxy competes with the same model load. Check first:

```bash
python3 - <<'PY'
from proxy.host_pressure import read_memory_snapshot, evaluate_pressure, describe_snapshot
s = read_memory_snapshot()
print(evaluate_pressure(s).level, "|", describe_snapshot(s))
PY
```

If the level is `warning` or `critical`, unload an idle model (or reduce local
context) before loading anything new. The pre-OOM alerts that fire here are
`HostMemoryAvailableLow*` and `GpuGttVsHostRamHigh*`
([monitoring/host_memory_alerts.yaml](../../monitoring/host_memory_alerts.yaml)).

## 3. Measure the outage you just ended

Every proxy start now logs the gap it closed, at startup, from the archived log
of the previous process:

```
WARNING Recovery: previous proxy log write 2026-11-02T12:02:45+00:00 (proxy.log.2026-11-02_120245), 650s before this start — the proxy was down until this start; confirm the supervisor (systemd Restart=always) is active ...
```

That figure is the measured recovery interval for an involuntary death. Record it
in the work item / incident note. With `Restart=always` + `RestartSec=2` the gap
for a supervised proxy is normally a few seconds; a ~10 minute gap means nothing
was supervising the process (the 2026-11-02 case: hand-launched, `RestartSec`
irrelevant because no unit existed).

```bash
# All recovery lines from the current log window
grep -h 'Recovery: previous proxy log write' proxy/logs/proxy.log*
```

## 4. Prevent a repeat

Protection ships in four layers; check each one after an incident.

| Layer | Mechanism | Verify |
|---|---|---|
| OOM ranking | proxy `oom_score_adj=-900` (system unit) / `0` (user), backend `+800` on the router **and** its model children | `server.oom_score_adj`, `server.llama_server_oom_score_adj`, unit `OOMScoreAdjust=`, `scripts/verify_oom_protection.py` |
| Supervision | `Restart=always`, `RestartSec=2` on the proxy and backend units | `systemctl --user cat llama-proxy.service` |
| Cgroup bound | `MemoryMax=4G` (proxy), `MemoryMax=70G` (backend) | `systemctl --user show -p MemoryMax llama-podman-wrapper.service` |
| Pre-OOM alert | `HostMemoryAvailableLow*`, `GpuGttVsHostRamHigh*` | `/alerts` in Prometheus, `monitoring/README.md` |

### OOM ranking — why the backend and not the proxy

The kernel ranks kill candidates by `oom_badness()`: resident memory as a
fraction of RAM, scaled by `(1000 + oom_score_adj) / 1000`. With the 2026-11-02
sizes on this host (124 GiB RAM, llama-server 17.1 GiB RSS, proxy 3.7 GiB):

| Process | `oom_score_adj` | badness |
|---|---|---|
| llama-server | 0 (before) | ~131 |
| proxy | 0 (before) | ~28 |
| llama-server | **+800** (after) | ~235 |
| proxy | **-900** (after, privileged) | ~2 |

So the backend is killed first and the proxy is last. `oom_badness` here is
`proxy.host_pressure.oom_badness` — the model is reproducible and unit-tested
against these numbers (`proxy/tests/test_host_pressure.py`), because actually
reproducing a global OOM on the live host would risk killing the proxy under
test.

#### Verify and reapply the backend ranking (`/proc`, LP-0MW3K7XJ30027RV6)

The score has to reach the **`llama-server` process**, not the start-script
wrapper the proxy spawns. Measured 2026-11-05 before the fix — the config asked
for `800` and every backend still read `0`:

```
$ for p in $(pgrep -x llama-server); do echo "$p oom_score_adj=$(cat /proc/$p/oom_score_adj)"; done
2089209 oom_score_adj=0     # router parent
2089312 oom_score_adj=0     # its per-model child
```

`start-llama.sh` runs `llama-server` as a foreground **child** and the kernel
copies `oom_score_adj` at fork time, so a write to the wrapper issued after that
fork was never inherited. The proxy now scores the whole process tree as its
spawn hook (before the quick-exit probe, so later forks inherit) and re-asserts
the ranking every ~30 s from the watchdog — which also covers the model children
the router forks when it loads a model, and an adopted backend that predates the
proxy.

Read the kernel's own values (raw `/proc`, exactly what the OOM killer uses):

```bash
# every llama-server process, router and model children alike
for p in $(pgrep -x llama-server); do
  printf '%s comm=%s oom_score_adj=%s\n' "$p" "$(cat /proc/$p/comm)" "$(cat /proc/$p/oom_score_adj)"
done
```

or use the verification tool, which walks the tree from the router (discovered by
listening port) and prints the same numbers plus a verdict:

```bash
python3 scripts/verify_oom_protection.py            # read-only, exit 1 on a gap
python3 scripts/verify_oom_protection.py --json     # machine-readable
python3 scripts/verify_oom_protection.py --apply    # reapply + re-verify
python3 scripts/verify_oom_protection.py --pid <wrapper-pid> --apply
```

`--apply` raises the score on every `llama-server` process it finds (and on the
root, so later children inherit); raising is always permitted, so it works
unprivileged and needs no restart.

**When the score cannot be applied (a recorded gap, not a silent pass).** The
proxy logs one WARNING per distinct gap, naming this remedy:

| Gap | Reason | Operator action |
|---|---|---|
| `no llama-server process in the tree of pid(s) …` | the backend is stopped, its start script has not forked it yet, or an adopted backend is undiscoverable (port unreadable) | run `scripts/verify_oom_protection.py`; if the backend is up and still not found, restart it so the proxy scores it at spawn time |
| `could not set oom_score_adj=… on pid …` | the process exited between discovery and the write (respawn race) | re-run `--apply`; the next router child inherits the router's raised score |
| proxy score `0` instead of `-900` | negative values need `CAP_SYS_RESOURCE` | install the **system** unit (`OOMScoreAdjust=-900`) or grant the capability; the backend ranking still applies |

The same values are logged at startup and exposed on `/metrics` as
`llama_oom_score_adj`; `proxy/tests/test_backend_oom_protection.py` pins the
router-child behaviour against real processes and real `/proc`.

Raising a score (`+800`) is always permitted. Lowering one below 0 needs
`CAP_SYS_RESOURCE`, which is why the negative proxy score is delivered by the
**system** unit (`OOMScoreAdjust=-900`, applied as root) and merely attempted at
runtime otherwise.

### Supervision

`scripts/systemd/llama-podman-wrapper.service` and
`scripts/systemd/llama-proxy.service` are the shipped units; both now use
`Restart=always` with `RestartSec=2`. A hand-launched proxy (plain
`start-proxy.sh`) is **not** supervised by design — it detaches and the launcher
exits. If you need unattended operation, install the units
([scripts/systemd/README.md](../../scripts/systemd/README.md)).

## 5. Failure modes this runbook does not cover

- **`constraint=CONSTRAINT_MEMCG`**: the kill was caused by a unit's own
  `MemoryMax`. Raise that unit's limit or reduce the model/context — the host
  itself was not out of memory.
- **A wedged (not killed) proxy**: `:8000` answers but requests hang. That is a
  different failure class — see [docs/systemd/llama-proxy.service](../systemd/llama-proxy.service)
  and the proxy wedge detection in `proxy/README.md`.
- **No `proxy.log` evidence at all**: before this item the launcher and the test
  suite shared `/tmp/proxy-startup.log`, and a pytest run at 12:03 overwrote the
  incident's real startup log. The launcher now writes a per-invocation file
  (`/tmp/proxy-startup-<UTC>-<pid>.log`) and keeps `proxy-startup.log` as a
  symlink to the newest, so old logs survive.
