# Proxy restart frequency: is ~24–34 restarts/day normal, and what causes them?

**Work item:** LP-0MV9UNKVK007TN7E
**Author:** Map (pi-agent)
**Analysis window:** 2026-10-09 06:06 → 2026-10-16 09:24 UTC (≈7.1 days), continuous rotated logs
**Status:** investigation complete; see follow-ups below.

## Verdict (TL;DR)

1. **The rate is high but not anomalous.** 170 restarts in 7.1 days ≈ **23.8/day**,
   consistent with the historical baseline (10-08=28 … 10-14=34). It is structurally
   driven, not a sporadic fault.
2. **98 % of restarts (167/170) are fast↔cheap mode-switch restarts.** Of these, 10
   align with the built-in schedule (`01:00 cheap`, `10:00 fast`); the remaining ~157 are
   activity/idle-driven switches issued by the ContextHub/herdr mode-switch worker (and
   occasional manual switches).
3. **There are no independent llama-server crashes.** All 44
   `watchdog detected llama-server exit code=0` events and 36/38
   `TTS watchdog: TTS server exited with code=0` events occur within seconds of a
   `Mode-switch restart spawned` line — they are **consequences** of the restart's
   `pkill`, not independent causes.
4. **3/170 restarts are unattributable** from the logs (2 TTS-watchdog-adjacent
   proxy-only restarts on 2026-10-14 00:26:05 and 00:31:40, and 1 fully unexplained at
   2026-10-14 22:34:49). See follow-up LP-0MVAPNC2900145RH.
5. **The single high-leverage fix is to make a mode switch non-destructive** (keep the
   running llama-server by equalising the fast/cheap slot pool, or restart only the
   proxy). That is already tracked by **LP-0MV9UNGEB009XUZ5**; flip-rate reduction is
   tracked by **LP-0MV9UNE75004903J**.

## Method and data

The proxy logs rotate into `proxy.log-<date>_00(.gz)` and `proxy.log.<date>_HH(.gz)` files.
Together they form a continuous record from 2026-10-09 06:06 to the present. Every process
start emits the marker `Starting LLama Proxy Server`; counting and classifying the ~90 s of
context before each marker yields the cause.

Reproducible commands (zgrep transparently handles both plain and gzipped files):

```bash
cd /var/log/llama-proxy

# 1. Total restarts in the retained window
zgrep -h "Starting LLama Proxy Server" proxy.log* | wc -l          # -> 170

# 2. Per-cause trigger counts
for p in "Mode-switch restart spawned" \
         "watchdog detected llama-server exit code" \
         "TTS watchdog: TTS server exited" \
         "Mode scheduler: enabled" \
         "Mode scheduler: applied scheduled mode"; do
  printf '%-48s ' "$p"; zgrep -h "$p" proxy.log* | wc -l
done
# -> Mode-switch restart spawned              167
# -> watchdog detected llama-server exit code  44
# -> TTS watchdog: TTS server exited           38
# -> Mode scheduler: enabled                  170
# -> Mode scheduler: applied scheduled mode     5

# 3. Restart with the lines that precede it (classification context)
zgrep -h -B3 "Starting LLama Proxy Server" proxy.log*
```

A one-pass classifier (walking the concatenated, timestamp-sorted log and looking back at
most 90 s or to the previous restart) attributes each restart to the *initiating* trigger,
preferring `Mode-switch restart spawned` over the downstream death/self-heal signatures.

## Per-cause classification (all 170 restarts)

| Class | Log signature (initiator) | Count | Share |
|---|---|---:|---:|
| **Mode switch** | `Mode-switch restart spawned: …/start-proxy.sh --restart` | **167** | **98.2 %** |
| TTS watchdog (downstream, unattributed) | `TTS watchdog: TTS server exited with code=0` (no preceding mode-switch spawn within 90 s) | 2 | 1.2 % |
| Unattributed | no trigger signature in the preceding 90 s | 1 | 0.6 % |
| llama-server death (independent) | `watchdog detected llama-server exit code=…` | 0 | 0 % |
| Proxy self-heal (independent) | `self-heal attempt n/3 started` | 0 | 0 % |

### Per calendar day

| Date (UTC) | Restarts | Mode-switch | TTS-adjacent | Unattributed |
|---|---:|---:|---:|---:|
| 2026-10-09 | 21 | 21 | 0 | 0 |
| 2026-10-10 | 16 | 16 | 0 | 0 |
| 2026-10-11 | 10 | 10 | 0 | 0 |
| 2026-10-12 | 32 | 32 | 0 | 0 |
| 2026-10-13 | 24 | 24 | 0 | 0 |
| 2026-10-14 | 34 | 31 | 2 | 1 |
| 2026-10-15 | 27 | 27 | 0 | 0 |
| 2026-10-16 (partial, ~9 h) | 6 | 6 | 0 | 0 |
| **Total** | **170** | **167** | **2** | **1** |

### Downstream (non-causal) events

These fire *after* a mode-switch restart and are misclassified as "causes" if counted naïvely:

- `watchdog detected llama-server exit code=0`: **44 events, 44/44** within 10 s of a
  `Mode-switch restart spawned`.
- `TTS watchdog: TTS server exited with code=0`: **38 events, 36/38** within 10 s of a
  mode-switch spawn. The 2 exceptions are the unattributed proxy-only restarts above.

Root cause: `proxy/scripts/start-proxy.sh --restart` runs
`pkill -f 'llama-server'` / `pkill -f 'qwentts'` (see
`proxy/scripts/start-proxy.sh`, "Phase 1/2/3" kill blocks). The still-running proxy's
watchdogs observe the intentional kill and log `exit code=0` + a self-heal attempt.

### Mode alternation

The mode at startup strictly alternates: of 170 consecutive starts, **166 are a change** of
mode (`fast` 86, `cheap` 84). This is the signature of an activity/idle oscillation, not a
stuck process:

- ContextHub/herdr `mode-switch-worker.ts` POSTs `{"mode":"fast"}` on operator command and
  `{"mode":"cheap"}` after ≥15 min idle (leader-only).
- The proxy enforces a 30-min fast→cheap cooldown (`MODE_SWITCH_COOLDOWN_SECONDS`), so each
  fast→cheap→fast cycle is ≥30 min. Median inter-switch gap is **32 min**, matching the
  cooldown — i.e. the rate is *bounded by design*, not runaway.

This also confirms the earlier flapping fix (LP-0MU5BOK3P0090O0V) is working: there are no
4-restarts-in-5-minutes episodes in the window.

### Trigger split (mode-switch restarts)

| Trigger | Count | Notes |
|---|---:|---|
| Built-in schedule (`01:00` cheap / `10:00` fast, ±10 min) | 10 | 2/day expected |
| ContextHub worker / manual activity-idle | ~157 | dominant |

## Is the rate unusual?

**No.** 23.8/day sits inside the historical range recorded in this work item
(10-08=28, 10-09=22, 10-10=16, 10-11=10, 10-12=32, 10-13=24, 10-14=34). The rate tracks
operator/agent activity: the proxy switches to `fast` when a command arrives and back to
`cheap` after 15 min idle, and each switch restarts the process. There is no evidence of a
runaway or a crash loop — the 30-min cooldown bounds the cadence.

## Blast radius of a restart

| Metric | Value |
|---|---|
| Restart → `llama-server is ready` | median **5 s**, max 5 s |
| Restart → first local (`routing_check`) request | median **5 s**, mean 7 s, max 105 s |
| Mode-switch drain outcome | 65 `timed_out=True` / 78 clean (143 drains) |
| Chat requests deferred during a drain | 372 |
| Restarts that preserved the warm llama-server (adopt) | **2 / 170** |
| Restarts that re-initialised llama-server | 168 / 170 |
| Local→remote fallback rate, 5 min before → after a restart | median 5.9/min → 9.2/min (**≈1.6×**) |
| Restarts with a ≥2× post-restart fallback spike | 77 / 170 (45 %) |

The dominant cost is **not** restart latency (5 s) but the loss of the warm KV cache and
the ~1.6× local→remote fallback spike afterwards (cold-prefill thundering herd). The
existing adopt path (`Adopted running llama-server … preserving warm prompt cache`) would
avoid this if the fast/cheap configs stopped differing in `session_slot_pool_size`
(fast = base 1, cheap = 3 — see `proxy/config-cheap.yaml`).

## Avoidable vs necessary, and prioritised fixes

| Cause | Necessary? | Fix | Tracking |
|---|---|---|---|
| Mode switch restarts (167) | No — avoidable | Make the switch non-destructive: equalise `session_slot_pool_size` and reuse the running llama-server (adopt path); or restart only the proxy | **LP-0MV9UNGEB009XUZ5** (slot-count evaluation) |
| High flip rate (~157 activity-driven) | Partly — some switching is desired | Reduce flips: longer idle threshold, hysteresis, suppress fast-on-command when already fast | **LP-0MV9UNE75004903J** |
| Downstream llama-server/TTS `exit code=0` watchdogs (82 events) | No — artefact | Disappear once restarts stop; additionally, quiesce watchdogs during an intentional restart to avoid duplicate TTS self-heal | covered by LP-0MV9UNGEB009XUZ5; see note below |
| 3 unattributable restarts | Unknown | Log the requesting actor/reason for every `set_mode` / `--restart` so restarts are always attributable | **LP-0MVAPNC2900145RH** (created) |
| Built-in schedule transitions (10) | Yes — intended | Keep; becomes cheap once switching is non-destructive | — |

### Decisions recorded (no new item)

- **Primary fix** is the non-destructive mode switch, already in scope of
  **LP-0MV9UNGEB009XUZ5**. No duplicate item created.
- **Flip-rate reduction** is already in scope of **LP-0MV9UNE75004903J**.
- **TTS exit-code=0** is not an independent fault: 36/38 events are caused by the restart's
  `pkill -f qwentts`. The existing TTS items (LP-0MRT928DD008136S, LP-0MS0Z9DTA003P6GB) are
  complete; no new TTS item is required, other than the watchdog-quiesce note above.

## Answers to the work-item questions

1. **Distribution over ≥7 days** — 98.2 % mode switch, 1.2 % TTS-watchdog-adjacent, 0.6 %
   unattributed; 0 % independent llama-server death. (Table above, per-day breakdown included.)
2. **Unusual vs baseline?** — No; 23.8/day is within the historical 10–34/day band and is
   a structural consequence of activity-gated mode switching.
3. **Avoidable vs necessary** — Almost all avoidable via non-destructive switching
   (already tracked); only the ~10 scheduled transitions per week are intended.
4. **Correlation with cooldown / fallback spikes** — Yes: post-restart local→remote
   fallback rate rises ≈1.6× and 45 % of restarts show a ≥2× spike, because the warm KV
   cache is discarded and llama-server is re-initialised on 168/170 restarts.
5. **Blast radius** — 5 s restart latency, but warm-cache loss and a fallback spike;
   143 drains with 65 timeouts and 372 deferred requests, so some in-flight work is lost
   when a drain exceeds the 30 s window.
