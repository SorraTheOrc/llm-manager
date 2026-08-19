# Qwen3 MTP (Multi-Token Prediction) Evaluation

Work item: LP-0MSNI1B68001VE6C · Status: **partial — MTP measurement blocked**

## Goal

Evaluate whether serving Qwen3.6-35B-A3B with llama.cpp's Multi-Token
Prediction (MTP) speculative decoding (`--spec-type draft-mtp`) yields a
measurable throughput improvement over the current non-MTP `Qwen3` serving,
without sacrificing response quality.

Expected gains per the llm.cpp MTP upstream work (PR #22673) and unsloth's
guide: ~1.5–2× generation speed at no accuracy loss, using the model's
built-in prediction heads to draft multiple tokens in parallel.

## Model configuration (committed)

A coexisting evaluation entry was added — production `plan`/`author`/`code`
chains are **unchanged** (still route local-first to `Qwen3`).

| Artifact | Change |
|----------|--------|
| `start-llama.sh` | New `qwen3-mtp` case: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_S` (Q4_K_S per constraints), `--spec-type draft-mtp --spec-draft-n-max 2`, single-slot (`-np 1`) noted |
| `models.ini` | New `[Qwen3-MTP]` router preset: `hf-repo = unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_S`, `ctx-size = 131072` (matches `[Qwen3]` for a fair A/B), q8_0 KV cache, flash-attn/swa-full/no-mmproj |
| `proxy/config.yaml`, `config-fast.yaml`, `config-cheap.yaml` | New `local-qwen3-mtp` model entry: first provider `llama_model: Qwen3-MTP` (local), same remote fallback chain as production (opencode → opencode-go → deepseek), aliases `local-qwen3-mtp` / `qwen3-mtp`, `tokenizer: qwen3` |

Target any benchmark/quality request at model `local-qwen3-mtp` or `qwen3-mtp`.

## Acceptance criteria status

| # | AC | Status |
|---|----|--------|
| 1 | Baseline measured (current non-MTP Qwen3.6) via `proxy/benchmarks/run_benchmark.py` | **Done** — see [Baseline results](#baseline-results) |
| 2 | MTP enabled (start-llama.sh + config.yaml) | **Done** — code committed |
| 3 | MTP throughput measured (TPS, acceptance rate, latency) | **BLOCKED** — needs MTP-capable llama-server build |
| 4 | Comparison documented (TPS factor, acceptance-rate trade-offs) | **Partial** — framework below; MTP numbers pending |
| 5 | Quality check (≥10 prompts, both configs, qualitative parity) | **BLOCKED** — same build dependency |
| 6 | Full test suite passes | **Done** — see test run |

## Blocker: local llama-server build lacks MTP support

The proxy's llama-server binary is

```
/home/rgardler/llama.cpp/build/bin/llama-server
version: 8782 (e97492369)   # built 2026-04-13
```

`--help` lists `--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]`
— **no `draft-mtp`**. MTP support was merged upstream in llama.cpp PR #22673
(commit `255582687 llama + spec: MTP Support (#22673)`, 2026-05-16), which
predates this build. The current checkout (`master` at `e97492369`) is a
descendant point **before** the MTP merge.

Verification performed:

- `llama-server --help` → no `draft-mtp` / `spec-draft-n-max` options.
- `common/arg.cpp` at `e97492369` → `--spec-type` whitelist has no MTP.
- MTP commit `255582687` is present in the local git object DB
  (`git cat-file`/`merge-base --is-ancestor` fails against `HEAD`), i.e. the
  upstream code exists but is not built.
- MTP-converted GGUF repo `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` exists on HF
  (Q4_K_S file present: `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`).

**Required to unblock AC3/AC4/AC5:** rebuild llama.cpp from a post-2026-05-16
master commit that includes PR #22673 and deploy it. The repository ships the
documented path:

```bash
scripts/rebuild-llama.sh
# clones https://github.com/ggml-org/llama.cpp.git (fresh, /tmp/llama_rebuild)
# cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON
# copies build/bin/llama-server -> /home/rgardler/llama.cpp/build/bin/llama-server
# (deploys over the binary the running proxy uses — requires a proxy restart)
```

Faster alternative: bump the existing checkout
(`/home/rgardler/llama.cpp`) to `origin/master` (already contains the MTP
commit) and rebuild incrementally in the configured `build/` dir
(`cmake --build build --config Release -j$(nproc)`), then deploy + restart.
Either path disrupts the live proxy; it is an operator decision.

> **Note:** MTP currently requires single-slot serving (`--parallel 1`). The
> production router runs `--parallel 3` (3 slots, per-slot ctx 43.7K). Deploying
> MTP therefore means swapping concurrency for per-request speed — the A/B must
> weigh TPS-per-request vs lost slot concurrency, and the proxy's
> `session_slot_pool_size`/slot_schedule would need to drop to 1 for a
> representative measurement.

## Baseline results (AC1)

Run: `python -m proxy.benchmarks.run_benchmark --baseline --model plan \
  --config models.ini --output benchmark-qwen3-baseline-20260817T183643Z.json`

- Tool: `proxy/benchmarks/run_benchmark.py` (proxy benchmark suite).
- Target: live proxy `http://localhost:8000`, model alias `plan`
  (routes local-first to **Qwen3** — non-MTP Qwen3.6-35B-A3B, Q5_K_M per
  `models.ini`, GPU offload ngl=80, 3 slots).
- Workload: 5 requests × the suite's standard 5 prompts, `max_tokens=128`,
  `concurrency=1`, `timeout=60s`.
- Routing: verified in proxy log — every benchmark request logged
  `[local] POST ... model=plan` → `provider=local-qwen3 model=Qwen3`
  (no remote fallback).

| Metric | Value |
|--------|-------|
| Requests completed | 5 / 5 (0 errors) |
| Avg total duration | 14.01 s |
| Avg tokens/sec | **11.06** |
| p95 tokens/sec | 16.42 |
| Avg TTFT (est.) | 0.16 s |
| p95 TTFT (est.) | 0.32 s |
| Total completion tokens | 640 |

Caveat: run captured **under concurrent live load** (≥3 herdr sessions
streaming large contexts at the same time; `available_slots=0` observed
mid-run; local_owner lease held by another session). The numbers therefore
reflect contended throughput, not a quiet benchmark. For the AC4 comparison,
re-run baseline and MTP **back-to-back under matched conditions** (ideally
low traffic) so the delta isolates the MTP effect.

### Re-run instructions (baseline and MTP)

```bash
# Baseline (current Qwen3):
python -m proxy.benchmarks.run_benchmark --baseline --model plan \
    --config models.ini --output baseline-qwen3.json

# MTP candidate (after unblock + proxy restart with MTP build):
python -m proxy.benchmarks.run_benchmark --candidate --model local-qwen3-mtp \
    --config models.ini --output candidate-qwen3-mtp.json

# Compare:
python -m proxy.benchmarks.compare_results baseline-qwen3.json candidate-qwen3-mtp.json
```

The MTP run must additionally record the **acceptance rate** from
llama-server (`/slots` or server logs expose draft-acceptance stats) — the
current `run_benchmark.py` output does not include it, so capture server-side
statistics during the run.

## Quality check (AC5) runbook

Run ≥10 representative prompts (mix of plan/author/code-style tasks) through
both `plan` (Qwen3) and `local-qwen3-mtp` and compare outputs for semantic
parity. Suggested prompt set: repo `proxy/benchmarks/prompts.json` (if
present) or the suite `DEFAULT_PROMPTS` + a few coding tasks. Document any
quality divergence (MTP hypothesis: none, per upstream claims).