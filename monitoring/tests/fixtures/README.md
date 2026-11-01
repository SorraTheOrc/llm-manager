# Recorded Prometheus API payloads

Fixtures for GPU-telemetry tests (LP-0MVT1NVAJ007KUGO, parent
LP-0MVNWUEY8008XM6W). They let the `gpu_utilisation` CLI
(LP-0MVT1O52G000LMC6) and the recording-rule work (LP-0MVT1O4O1004ND6L) be
tested against **recorded `/api/v1/query_range` responses** instead of a live
Prometheus, so the tests are deterministic, offline and free of scrape-timing
flakiness.

## File layout

Every file has the same two top-level keys:

```json
{
  "capture": {
    "endpoint":  "http://127.0.0.1:9090/api/v1/query_range",
    "query":     "<PromQL>",
    "window":    {"start": "...", "end": "...", "start_unix": 0, "end_unix": 0,
                  "duration_seconds": 7200},
    "step_seconds": 15,
    "captured_at": "...",
    "source":  "captured live ..." | "authored" | "mixed",
    "notes":   "<how the values were produced>"
  },
  "response": {"status": "success", "data": {"resultType": "matrix", "result": [...]}}
}
```

`window_with_gap.json` is the exception: it carries `responses`, a mapping of
PromQL → the same `response` shape, so one test can serve every query the CLI
issues for a single window from one file.

| File | Query | Source |
|---|---|---|
| `gpu_computing_bool.json` | `gpu:computing:bool` | captured live |
| `gpu_power_watts_amdgpu.json` | `gpu:power_watts:amdgpu` | captured live |
| `up_llama_proxy.json` | `up{job="llama-proxy"}` | captured live |
| `up_node_exporter.json` | `up{job="node-exporter"}` | captured live |
| `llama_slots_busy.json` | `llama_slots_busy` | authored (series not published yet) |
| `llama_slots_sample_timestamp_seconds.json` | `llama_slots_sample_timestamp_seconds` | authored |
| `gpu_computing_workload_bool.json` | `gpu:computing_workload:bool` | authored |
| `window_with_gap.json` | all of the above | mixed |

## How the live captures were made

One `query_range` call per query, 15 s step (the deployed
`scrape_interval`/`evaluation_interval`), over a two-hour window:

```bash
start=$(( $(date +%s) / 15 * 15 - 7200 ))
end=$(( start + 7200 ))
curl -sG 'http://127.0.0.1:9090/api/v1/query_range' \
  --data-urlencode 'query=gpu:computing:bool' \
  --data-urlencode "start=$start" --data-urlencode "end=$end" \
  --data-urlencode 'step=15'
```

The power, `gpu:computing:bool` and `up` series exist on this host, so those
payloads are **real and untouched**: the timestamps, metric label sets
(`instance="localhost:9100"` for node-exporter, `instance="localhost:8000"` for
the proxy) and values are exactly what Prometheus returned. That matters for
the label-matching trap the recording-rule child has to respect: the two input
series carry *different* `instance` labels.

## How the authored payloads were made

`llama_slots_busy`, `llama_slots_sample_timestamp_seconds` and
`gpu:computing_workload:bool` are not published on this host yet — they are
delivered by sibling children LP-0MVT1O0VG00905EL (sampler) and
LP-0MVT1O4O1004ND6L (recording rule). Their payloads are authored to the
documented `/api/v1/query_range` response shape over the *same* real window and
step, derived from the captured power series:

- `llama_slots_busy` = 2 where the captured `gpu:computing:bool` is 1, else 0
  (a computing GPU has slots occupied; an idle one does not);
- `llama_slots_sample_timestamp_seconds` = the sample timestamp itself, i.e. a
  heartbeat that proves the sampler was alive at that instant;
- `gpu:computing_workload:bool` = 1 only where power ≥ 90 W **and**
  `llama_slots_busy` ≥ 1.

Do not over-fit tests to these values: they exist to pin the *shape* and the
gap semantics, not to be a golden readout of real hardware behaviour. When the
sibling children land, re-capture the real series with the command above and
replace the authored files.

## The deliberate gap

Every authored stream carries the same hole: **40 consecutive missing samples
(600 s at a 15 s step)** at 65 minutes into the two-hour window
(`window.start + 3900 s` … `+ 4500 s`). `window_with_gap.json` clips the live
series to the same hole so the whole bundle models a window in which the host
was not sampled at all.

That is the case the parent brief forbids getting wrong: a window with missing
samples must report the missing share as **unknown**, never as idle and never
renormalised away. Over this window the expected coverage is 481 samples,
441 are present, so the honest unknown share is 40/481 = **8.316 %** — the
exact figure the CLI integration test asserts.
