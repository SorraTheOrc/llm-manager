
## How to add a new model to the proxy

This document describes how to add models, aliases, fallback chains and custom system prompts to the proxy and the proxy managed llama-server 
for locally hosted models.

Note that once you have added models/aliases to the Proxy you will need to instruct your agent framework to use them. How to do this
is outside the scope of this document.

## Adding the model to the local start script

- When running a local model via the proxy managed llama-server instance the proxy expects models to be startable by the repository `start-llama.sh` script (`/home/rgardler/projects/llm/start-llama.sh`). To make a new local model available via that script you must add a case block for the model name to the script.
 - Steps to add your model to `start-llama.sh`:

 ### 1) Register the model preset for router-mode (models.ini)

 - If you run llama-server in router mode (the proxy talks to a router process on a fixed port), you must also register a preset in the router's `models.ini` so the router knows how to start/download the model. Add a new section using the exact model id the router will expose (example below). After editing, restart or reload the router process so it re-reads `models.ini`.

 Example `models.ini` snippet:

 ```ini
 [ggml-org/gemma-4-31B-it-GGUF:Q8_0]
 hf-repo = ggml-org/gemma-4-31B-it-GGUF:Q8_0
 ctx-size = 262144
 # Optional: embeddings = true, pooling = mean, etc.
 ```

 Verify the router knows the model:

 ```bash
 curl -sS http://localhost:8080/models | jq .
 curl -sS -X POST http://localhost:8080/models/load -H "Content-Type: application/json" -d '{"model":"ggml-org/gemma-4-31B-it-GGUF:Q8_0"}' -v
 ```

  1. Pick the canonical model name that will be used in `proxy/config.yaml` `llama_model:` and in requests to the proxy. The `start-llama.sh` script lowercases the first CLI argument, so use a lowercase name in the case pattern.
  2. In `start-llama.sh` add a `case "$model" in` entry (follow the existing examples `qwen3`, `qwen2.5`, `gpt120`). At minimum set these variables inside the block: `REPOID`, `MODEL`, `QUANTIZATION`, `CONTEXT`, `BATCH_SIZE`, and any of `CHAT_TEMPLATE_KWARGS`, `REASONING_FORMAT`, `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `EXTRA_CMD_SWITCHES` as needed for the model.
     - Example block (copy and adapt fields as necessary):

```bash
  mymodel)
    REPOID=Vendor
    MODEL=MyModel-Name-GGUF
    QUANTIZATION=Q5_K_M
    CONTEXT=131072
    BATCH_SIZE=512

    CHAT_TEMPLATE_KWARGS='{"reasoning_effort": "medium"}'
    REASONING_FORMAT=none

    TEMP=0.7
    TOP_P=1.0
    TOP_K=40
    MIN_P=0

    EXTRA_CMD_SWITCHES="--jinja"
    ;;
```

  3. Update the final `*)` default section of the script (if appropriate) to include the new model name in the recognised list shown to users (the script prints the supported models when an unrecognised name is provided).
  4. If you maintain a container image that packages `start-llama.sh` (see `proxy/container/Containerfile` and `proxy/CONTAINERS.md`), update any documentation or container build steps if the model requires additional artifacts or mount points.
  5. Test locally by running the script directly from repo root, e.g. `/home/rgardler/projects/llm/start-llama.sh mymodel` and ensure the llama-server process starts and exposes embeddings (if required) on the configured port.

### Making the model downloadable from Hugging Face

Intro

Regardless of whether the proxy is running in router mode or single-model mode, the model files should come from a canonical source such as Hugging Face. The easiest and most reliable approach is to let the llama-server perform the download. To do this:

- Add the new model mapping to `start-llama.sh` (see steps below).
- Start the llama-server invoking the start script with the model name, for example:

  ```bash
  ~/projects/llm/start-llama.sh <model_name>
  ```

  The start script will construct the `-hf` argument for `llama-server` and the server will fetch the model from Hugging Face, storing it in the configured models location. When the download completes you can stop the server and run the proxy normally; the proxy will use the downloaded model whether it runs in router or non-router mode.

What you must change to make `your_model` download from Hugging Face:

- If you run the start script (non-router fallback):
  1. Add a `your_model)` case block in `start-llama.sh` matching the lowercase model name (the script lowercases the arg).
  2. Set `REPOID`, `MODEL`, `QUANTIZATION`, etc., inside that block so the script builds the correct `-hf` value for `llama-server`.
  3. Example: create a block that sets `REPOID=owner`, `MODEL=your_model-gguf`, `QUANTIZATION=Q8_0`, then llama-server gets `-hf "owner/gemma4-gguf:Q8_0"`.

Notes:
- The proxy passes the configured `llama_model` value through to the local start script. Keep the name consistent between `proxy/config.yaml` `llama_model:` and the `case` pattern in `start-llama.sh`.
- The script lowercases the first argument before matching; use lowercase in your `case` pattern to avoid mismatches.

## Model configuration

## Provider fallback configuration

Models can define an ordered list of `providers` for automatic failover. The `providers` list replaces the old top-level `endpoint`/`api_key_env`/`headers` fields. This is a **breaking change** from the old flat format.

### Provider entry fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique identifier for this provider entry |
| `type` | string | yes | `"local"` or `"remote"` |
| `endpoint` | string | remote / local (optional) | Base URL of the provider API. For `remote`, required. For `local`, optional: points at a pre-existing llama-server instance (LP-0MRPILSMW004T4H8). |
| `api_key_env` | string | remote | Environment variable containing the API key |
| `headers` | dict | remote (optional) | Additional headers to include |
| `llama_model` | string | local | Name of the local model |
| `rate_limit_rpm` | int | optional | Maximum dispatches to this entry per rolling **60-second** window (LP-0MVDQ1GMO00801PL). Intended for free tiers with a per-minute allowance (e.g. `munitoai-free`). Default: unset/`0` = ungated. |
| `budget` | dict | optional | Per-**entry** behaviour override for the free-tier budget knobs (`reserve_ratio`, `wait_max_seconds`, `max_waiters`, `affinity_ttl_seconds`) — precedence: entry > model > server > default (LP-0MVDVK5TM004TQQV; see [Per-model and per-entry budget knobs](#per-model-and-per-entry-budget-knobs-budget)). |
| `max_context_tokens` | int | optional | Maximum **estimated prompt context** the entry may serve (LP-0MVF9N2QL001CDA7). A request whose routing estimate exceeds this ceiling SKIPS the entry and falls through to the next eligible provider — a size mismatch, not a failure (no cooldown/affinity/usage-limit/queue side effects). Absent, `0` or non-positive disables the gate (byte-for-byte previous behaviour). Shipped `config.yaml` values are sourced (upstream context window / provider per-minute token allowance) and recorded in a config comment; see [Per-entry context limits](#per-entry-context-limits-max_context_tokens). Applies to local entries too: composes as an additional hard cap (`min(existing local thresholds, max_context_tokens)`). |
| `display_name` | string | optional | Human-readable tier label for the X-Resolved-Model header (e.g. `MunitoAI Free`, `OpenCode Go 3`). Overrides the `provider` brand in the display string only — failure-domain, cooldown and usage-limit keying still use `provider`. Use it to distinguish same-brand tiers/accounts (LP-0MVCQ36QV009QX8K, LP-0MVDUW7OB006UT62). |
| `priorities` | list | optional | The `low`/`medium`/`high` priorities this entry may serve (LP-0MVGPZWY0001JD1D). The model's priority chain is its ordered `providers` list filtered by the request's effective priority, so membership is declared here and nowhere else. Absent on **every** provider of a model = unlabelled model (all priorities served by all providers). Validated at load; see [Priority-chain membership](#priority-chain-membership-priorities). |

#### Priority-chain membership (`priorities`)

Priority routing is declared once, on the provider definition itself. The
chain for a request is the model's own ordered `providers` list filtered by
the request's effective priority (`X-Priority`, capped to `low` inside the
cheap window):

```text
chain(P) = [p for p in providers if P in p["priorities"]]   # providers order
```

```yaml
providers:
  - name: local-qwen3
    priorities: [low, medium, high]   # served for every priority
    type: local
    llama_model: Qwen3
  - name: munitoai
    priorities: [medium, high]        # never in the ``low`` chain
    type: remote
```

Rules (each one is a load-time `ValueError` from `validate_priority_config`, so
nothing fails silently at 02:00 instead of at startup):

- **No tier implication:** `[medium, high]` does not include `low`. Membership
  is exactly the list written.
- **Model-level opt-in, fail-closed:** if any provider of a model declares
  `priorities`, every provider of that model must, and together they must
  cover `low`, `medium` and `high`.
- **Shape checks:** the value must be a non-empty list of known priority names
  with no duplicates.
- **No silent widening:** a labelled model's filtered chain is used as-is; an
  empty result never expands to the full `providers` list.
- **Per-priority ordering is not supported:** chain order is always the
  `providers` order. A provider that must lead one priority and trail another
  needs two entries (or two models), so this is a price worth understanding
  before adding a paid tier to a `low` chain.

The former per-model `priority_routes` name map is **removed**. A config that
still declares it fails at load; see the README section
[Migrating from `priority_routes`](README.md#migrating-from-priority_routes).

#### Per-entry context limits (`max_context_tokens`)

A rate-limited tier whose allowance is expressed per minute (MunitoAI
advertises `x-ratelimit-limit-*` headers) can have its whole per-minute input
budget spent by a single oversized prompt. `max_context_tokens` caps the
**estimated prompt context** a request may carry before the entry is skipped:

- the estimate reuses the authoritative routing pipeline
  (`_get_tokenizer_for_model` + `_estimate_effective_prompt_tokens_for_routing`
  + `token_estimate_multiplier`), so the gate is consistent with the local
  thresholds and the hard routing cap, and it is computed **once per request**
  (no duplicate tokenizer pass per provider);
- a skip is recorded as an attempt with status `provider_context_exceeded`
  (carrying the estimate and the limit) and logged with provider, estimate,
  limit and reason — it is **not** a failure, so no cooldown, failure-domain
  exclusion, usage-limit reset, affinity record or contention-queue entry is
  applied, and the entry stays eligible for later, smaller requests; each skip
  also increments the cumulative `backend_signals.provider_context_exceeded`
  counter (mirroring `context_too_large`; read it from `/health` or
  `/admin/metrics`);
- at least one tier is left effectively ungated (capped at the largest
  upstream context window), so an oversized prompt always has a destination
  and the chain cannot exhaust because of caps alone.

Shipped values in `proxy/config.yaml` (probed 2026-10-19, each sourced in the
config comment next to the entry):

| Entry | `max_context_tokens` | Source |
|-------|----------------------|--------|
| `munitoai-free` | 150000 | MunitoAI free-tier `x-ratelimit-limit-input-tokens` (rolling 60 s input allowance); below the 1,048,576-token upstream window |
| `munitoai` | 400000 | MunitoAI paid-tier `x-ratelimit-limit-input-tokens` (rolling 60 s input allowance) |
| `opencode-go*` | 1000000 | Model-registry context window for `opencode-go/deepseek-v4.1-flash` (OpenCode Go publishes plan quota as percentages, not per-minute tokens) |
| `deepseek-flash` | 1048576 | `api.deepseek.com` `/models` `context_window` — the largest window in the chain (effectively ungated destination) |
| `compact` chain | 1048576 | model-registry / `api.deepseek.com` windows (documentation-consistency caps) |

Copy-pasteable example — a capped local entry, a capped free tier and an
ungated final tier, so an oversized prompt always has a destination:

```yaml
models:
  some-model:
    providers:
      - name: local-qwen3
        type: local
        llama_model: Qwen3
        max_context_tokens: 50000      # extra hard cap on top of the local thresholds
      - name: some-free-tier
        type: remote
        endpoint: https://api.example.com/v1
        api_key_env: SOME_FREE_KEY
        model: some-model
        max_context_tokens: 150000     # estimated prompt-context ceiling
      - name: some-final-tier
        type: remote
        endpoint: https://api.example.com/v1
        api_key_env: SOME_FINAL_KEY
        model: some-model
        # max_context_tokens omitted -> ungated (absent/0/non-positive)
```

#### Free-tier request budgets (`rate_limit_rpm`)

A free tier that allows only a handful of requests per minute is exhausted
almost immediately when several agents share it, and blindly dispatching into
it produces upstream `429`s and long stalls before the chain falls through to
a paid tier. Setting `rate_limit_rpm` on such an entry makes dispatch
capacity-aware:

- while the rolling 60-second window has capacity, the entry is used normally;
- a fraction of the window (`server.provider_rate_limit_reserve_ratio`, default
  `0.3`) is reserved for sessions already using the entry, so a request from a
  **new** session may use at most `rate_limit_rpm - reserve` dispatches and then
  skips the entry **immediately** — it is never parked waiting, so waiting
  requests cannot compound the contention;
- a session that was already served by the entry (affinity recorded on a 2xx
  dispatch) may use the reserved slots and, if the window is still full, wait a
  short bounded time for it to reset (`server.provider_rate_limit_wait_max_seconds`,
  default 15 s), keeping the same upstream provider and its prompt cache/prefix
  continuity. Waiting is capped in concurrency by
  `server.provider_rate_limit_wait_max_waiters` and aborted as soon as the
  client disconnects;
- affinity expires after `server.provider_affinity_ttl_seconds`, so a session
  returning much later counts as new again.

```yaml
      - name: some-free-tier
        type: remote
        endpoint: https://api.example.com/v1
        api_key_env: SOME_FREE_KEY
        forward_session_headers: false
        model: some-model
        rate_limit_rpm: 15
```

Tuning knobs (all optional, precedence is **entry `budget` > `models.<name>.budget` > `server.*` > default**): `provider_rate_limit_reserve_ratio` (share
of the window reserved for in-progress sessions, default 0.3),
`provider_rate_limit_wait_max_seconds` (default 15),
`provider_rate_limit_wait_max_waiters` (default 8) and
`provider_affinity_ttl_seconds` (default 900). A model block tunes a whole
chain; an entry `budget` block tunes one entry:

```yaml
models:
  plan:
    budget:
      reserve_ratio: 0.4
      wait_max_seconds: 10
      max_waiters: 4
      affinity_ttl_seconds: 600
    providers:
      - name: some-free-tier
        type: remote
        endpoint: https://api.example.com/v1
        api_key_env: SOME_FREE_KEY
        rate_limit_rpm: 15
        budget:
          reserve_ratio: 0.2
```

All keys are optional at every layer; absent keys fall through to the next
layer, so existing configs behave exactly as before (no migration). The budget
**bucket stays keyed by provider entry**: models sharing an entry share its
`rate_limit_rpm` window — the blocks change how each chain treats that window,
never its size. A skip/wait log names the winning layer as
`budget_source=<knob>@<layer>` (only when a model/entry block contributed).
Note: `models.compact` is not budget-gated — the summarizer dispatches to its
providers directly (LP-0MVFYYYXS0011T58).

Skip decisions are logged as `Provider budget exhausted: skipping provider=…`
(with limit, remaining and reason) and recorded in the request diagnostics as
an attempt with `status=rate_limited`; an in-progress session that waited is
recorded as `status=rate_limited_waited` with its `elapsed_wait_seconds`.

### Multiple local backends (LP-0MRPILSMW004T4H8)

Each `type: local` provider may declare its own `endpoint` URL (host:port)
pointing at a pre-existing llama-server instance. Listing several local
providers with different endpoints pools their GPU slot capacity: the proxy
routes to the first local provider and falls through to the next when it is
busy or unhealthy (sequential fallback, no load balancing).

- Omitting `endpoint` keeps the legacy behaviour — the provider routes to
  `http://localhost:{server.llama_server_port}`.
- Each server tracks its own dispatch leases (keyed per-endpoint) and
  persists its slot snapshots in its own `{session_slot_save_path}/{host}-{port}/`
  subdirectory.
- The proxy assumes the servers are already running; lifecycle management
  is out of scope for multi-backend setups.

```yaml
models:
  qwen3:
    providers:
      - name: local-qwen3-server-1
        type: local
        llama_model: Qwen3
        endpoint: http://192.168.0.199:8080
      - name: local-qwen3-server-2
        type: local
        llama_model: Qwen3
        endpoint: http://192.168.0.200:8080
      - name: remote-fallback
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
    aliases:
      - qwen3*
```

### Example: Local model with remote fallback

```yaml
models:
  plan:
    providers:
      - name: local-qwen3
        type: local
        llama_model: qwen3
      - name: primary_remote
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
      - name: secondary_remote
        type: remote
        endpoint: https://api.provider-b.com/v1
        api_key_env: PROVIDER_B_KEY
    aliases:
      - plan*
      - map
```

## Custom system prompts per model / alias

Each model entry may optionally include a `system_prompt` configuration to inject a default
system prompt into all requests targeting that model or any of its aliases. This is useful
for setting consistent agent personalities (e.g. "You are a helpful coding assistant")
without requiring clients to send the prompt.

### Configuration

```yaml
models:
  assistant-model:
    type: local
    llama_model: gemma4
    aliases:
      - "assistant"
      - "asst"
    system_prompt:
      mode: "prepend"       # "override" or "prepend" (required)
      file: "proxy/prompts/assistant.txt"  # path relative to repo root
```

### `mode` field (required)

| Mode | Behaviour |
|------|-----------|
| `override` | Replaces **all** client-supplied system messages with the prompt file content. |
| `prepend`  | Inserts the prompt file content as the first system message **before** any client-supplied system messages. |

If `system_prompt` is present but `mode` is missing or invalid, config validation fails at startup.

### `file` field (required)

Path to the prompt text file. This can be:
- A **relative path** - resolved against the repository root (e.g. `proxy/prompts/assistant.txt`)
- An **absolute path** - used as-is

### Prompt resolution precedence

When a request arrives, the proxy resolves the prompt file in this order:

1. **Local override**: `.sorraAgents/prompts/<alias>.txt` (project-local, not committed)
2. **Repo default**: the path specified in `system_prompt.file`
3. **No prompt** if none of the above exist

This allows operators to deploy per-instance prompt overrides without modifying the repository.

### File format and limits

- Files must be **plain text UTF-8**.
- Maximum file size: **64 KB**. Files larger than this are ignored and logged.
- Files with non-UTF-8 content are ignored and logged.
- No caching: files are read on every request (safe for rapid iteration).

### Security considerations

- **Do not store secrets or credentials in prompt files.** Prompts are plaintext and may
  appear in logs, debug endpoints, or upstream request payloads.
- Local override files (`.sorraAgents/prompts/`) are outside the repository and should
  be managed via deployment tooling, not committed.
- The debug endpoint (see admin endpoints) returns prompt previews; enable it only
  for development or QA.

### Example prompt file

Create `proxy/prompts/assistant.txt`:

```
You are a helpful and knowledgeable assistant. Your responses should be
concise, accurate, and well-structured. When asked to write code, include
comments and follow best practices.
```

### Worked example: the `author` model (first live deployment)

The production `proxy/config.yaml` wires the wiki authoring guidelines into the
`author` model via a `system_prompt` block — this is the **first live use** of
the feature:

```yaml
author:
  providers:
    # ... existing providers unchanged ...
  system_prompt:
    mode: "override"        # replaces any client-supplied system messages
    file: "proxy/prompts/author.txt"
```

- The proxy auto-loads `proxy/prompts/author.txt` (the mandatory wiki
authoring rules: openness terminology, citation, and "explain names on first
mention") for every `model=author` request, so no client needs to paste the
guidelines.
- `override` was chosen because the authoring client sends no client-side system
prompt — author.txt is intended as the complete system prompt for wiki
authoring. If a client later sends its own system prompt it will be replaced.
- Resolution precedence still applies: a per-deployment local override at
`.sorraAgents/prompts/author.txt` (uncommitted) takes precedence over the
committed default.
- Wiring: LP-0MV5S5IMX0091DZ5; prompt content owner: LP-0MUYJHQ2W002I0CE
(do not change author.txt content here).

## Alias resolution

- The proxy resolves aliases using `get_model_config()` in `proxy/lifecycle.py`.
  See that function for exact precedence rules (exact match > case-insensitive exact
  alias > wildcard pattern).
- **Friendly aliases** like `plan`, `code`, or `embeddings` can be added to any
  model's `aliases` list. They behave identically to regular aliases and follow
  the same case-insensitive matching and precedence rules.
- The `/v1/models` endpoint automatically lists all configured aliases in the
  `aliases` field of each model entry, making friendly names discoverable.

## Routing to local backends

 - Ensure the model's `type` value is consulted when routing. For `type: local` entries, use the existing `proxy_to_local(request, path)` helper and pass the configured `llama_model`. When the local provider declares an `endpoint`, pass it as the third argument so the request routes to that specific llama-server (`proxy_to_local(request, path, endpoint)`; `None`/omitted falls back to `localhost:{llama_server_port}` — see LP-0MRPILSMW004T4H8). For `type: remote` entries, use `proxy_to_remote` and the configured `endpoint`/`api_key_env`.

## Native tokenizer (optional, LP-0MSEQ71IF0003FRT)

 - When a local model's tokenizer differs from tiktoken's cl100k encoding (e.g. Qwen3 undercounts ~1.69x for dense prose), add a `tokenizer: <name>` field to the model entry so routing/persistence estimates use exact native counts:

   ```yaml
   plan:
     tokenizer: qwen3   # named reference into the lazy registry
     providers:
       - name: local-qwen3
         type: local
         llama_model: Qwen3
   ```

 - Add a matching registry entry in `proxy/proxy/tokenizers.py` (`TOKENIZER_REGISTRY`) pointing at a vendored `tokenizer.json`, and vendor the file via `scripts/vendor_qwen3_tokenizer.sh` (checksum-verified; committed to the repo, no runtime network needed).
 - With a native tokenizer active the `token_estimate_multiplier` is forced to 1.0 for that model; models without a `tokenizer` field keep the tiktoken + multiplier fallback. See `docs/llama-router.md`.

## Capability validation

- The proxy does not require or use a `supports` field. The proxy forwards requests to the configured backend
  and relies on the backend (llama-server or remote provider) to accept or reject the request. If the backend
  returns an error indicating the model does not support embeddings, the proxy should surface that error to callers.

## Integration tests

- Create an integration test under `proxy/tests/test_embeddings_integration.py` that:
  - Spins up the proxy (or uses a test harness) and a test llama-server hosting `mxbai-embed-large-v1`.
  - POSTs to `/v1/embeddings` with `model: "embeddings"` and `input: "test string"`.
- Validates the response follows the OpenAI embeddings format and that the returned vector length equals the documented dimension.

## Troubleshooting

- If the proxy returns 404 for a model: check the model config and alias mapping.
- If the embeddings response is empty or the wrong dimension: verify the llama-server exposes the expected embedding dimension and that the model supports embeddings.
- If a remote provider returns HTTP 500 despite working when called directly via curl:
  - Check whether the provider is receiving unexpected headers from the proxy.
  - The most common culprit is the `session_id` header, which certain providers
    (notably OpenCode's endpoints) reject with HTTP 500. The proxy forwards
    session affinity headers (`session_id`, `x-client-request-id`,
    `x-session-affinity`) by default for session locality, but some upstreams
    do not support them.
  - **Fix**: Add `forward_session_headers: false` to the remote provider config
    in the model's `providers` list. This strips session headers before
    forwarding the request upstream.
    ```yaml
    - name: my-remote-provider
      type: remote
      endpoint: https://api.example.com/v1
      model: my-model
      forward_session_headers: false   # <-- add this
    ```
  - To verify, check the proxy log for `[remote] upstream error status=500`
    for the provider, and confirm the `session_id` header is being stripped
    after adding the config key.

## Worked example (mxbai-embed-large-v1)

- Add the example config entry above to the `models:` section of `proxy/config.yaml`.
- Add alias mapping code in `get_model_config()`.
- Add the integration test `proxy/tests/test_embeddings_integration.py`.

## References
- Parent work item: LP-0MN557XBD0H8B8PC - Add an embeddings specific model
