
## How to add a new model to the proxy

This document describes the checklist and a worked example for adding a new model to the proxy. It is intended for engineers who need to add small local or remote models (including embeddings models) to the proxy.

## Adding the model to the local start script

- When running a local fallback the proxy expects models to be startable by the repository `start-llama.sh` script (`/home/rgardler/projects/llm/start-llama.sh`). To make a new local model available via that script you must add a case block for the model name to the script.
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

### Making gemma4 downloadable from Hugging Face

Intro

Regardless of whether the proxy is running in router mode or single-model mode, the model files should come from a canonical source such as Hugging Face. The easiest and most reliable approach is to let the llama-server perform the download. To do this:

- Add the new model mapping to `start-llama.sh` (see steps below).
- Start the llama-server inside the container (distrobox) and invoke the start script with the model name, for example:

  ```bash
  distrobox enter llama -- ~/projects/llm/start-llama.sh <model_name>
  ```

  The start script will construct the `-hf` argument for `llama-server` and the server will fetch the model from Hugging Face, storing it in the configured models location. When the download completes you can `exit` the container and run the proxy normally; the proxy will use the downloaded model whether it runs in router or non-router mode.

What you must change to make `gemma4` download from Hugging Face:

- If you run the start script (non-router fallback):
  1. Add a `gemma4)` case block in `start-llama.sh` matching the lowercase model name (the script lowercases the arg).
  2. Set `REPOID`, `MODEL`, `QUANTIZATION`, etc., inside that block so the script builds the correct `-hf` value for `llama-server`.
  3. Example: create a block that sets `REPOID=owner`, `MODEL=gemma4-gguf`, `QUANTIZATION=Q8_0`, then llama-server gets `-hf "owner/gemma4-gguf:Q8_0"`.

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
| `endpoint` | string | remote | Base URL of the remote API |
| `api_key_env` | string | remote | Environment variable containing the API key |
| `headers` | dict | remote (optional) | Additional headers to include |
| `llama_model` | string | local | Name of the local model |

### Example: Local model without fallback

```yaml
models:
  mxbai-embed-large-v1:
    providers:
      - name: local-embed
        type: local
        llama_model: mxbai-embed-large-v1
    aliases:
      - embeddings
```

### Example: Remote model with fallback

```yaml
models:
  my-remote-model:
    providers:
      - name: primary
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
      - name: fallback
        type: remote
        endpoint: https://api.provider-b.com/v1
        api_key_env: PROVIDER_B_KEY
    aliases:
      - my-model*
```

### Migration guide from old format

**Before (old flat format - deprecated):**
```yaml
models:
  my-model:
    type: remote
    endpoint: https://api.example.com/v1
    api_key_env: EXAMPLE_API_KEY
    aliases:
      - my-*
```

**After (new providers list - required):**
```yaml
models:
  my-model:
    providers:
      - name: my-provider
        type: remote
        endpoint: https://api.example.com/v1
        api_key_env: EXAMPLE_API_KEY
    aliases:
      - my-*
```

**Note:** The `type` field is now specified inside each provider entry, not at the model level. The old top-level `endpoint`/`api_key_env`/`headers` format is deprecated and will be removed in a future release.

### Adding a new model

 - Add an entry under the `models:` section of `proxy/config.yaml`. Required/typical fields (follow existing config style):
   - `providers`: a list of provider configs (see above)
   - For local providers: `type: local`, `llama_model`
   - For remote providers: `type: remote`, `endpoint`, `api_key_env` (and optional `headers`)
   - `aliases`: optional list of aliases (e.g. `embeddings`)

**Friendly model aliases:** You can use short, human-friendly names like `plan`
or `code` as aliases. These are listed in the `aliases` array alongside regular
model IDs. Callers can then pass `model: "plan"` or `model: "code"` in API
requests without needing to know the internal preset ID. See `README.md` for
full alias resolution rules (case-insensitive matching, wildcard precedence).

Example (add under `models:` in `proxy/config.yaml`):

```yaml
models:
  mxbai-embed-large-v1:
    providers:
      - name: local-embed
        type: local
        llama_model: mxbai-embed-large-v1
    aliases:
      - "embeddings"
    # Note: do not include a `supports` field here - the proxy will forward requests and the backend
    # (llama-server or remote provider) will surface errors if a model does not support a capability.

  # Remote example
  example-remote:
    type: "remote"
    endpoint: "https://models.example.com"
    api_key_env: "EXAMPLE_API_KEY"
    aliases:
      - "example-*"
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

 - Ensure the model's `type` value is consulted when routing. For `type: local` entries, use the existing `proxy_to_local(request, path)` helper and pass the configured `llama_model`. For `type: remote` entries, use `proxy_to_remote` and the configured `endpoint`/`api_key_env`.

## Capability validation

- The proxy does not require or use a `supports` field. The proxy forwards requests to the configured backend
  and relies on the backend (llama-server or remote provider) to accept or reject the request. If the backend
  returns an error indicating the model does not support embeddings, the proxy should surface that error to callers.

## Integration tests

- Create an integration test under `proxy/tests/test_embeddings_integration.py` that:
  - Spins up the proxy (or uses a test harness) and a test llama-server hosting `mxbai-embed-large-v1`.
  - POSTs to `/v1/embeddings` with `model: "embeddings"` and `input: "test string"`.
- Validates the response follows the OpenAI embeddings format and that the returned vector length equals the documented dimension.

## Adding the model to the local start script

- When running a local fallback the proxy expects models to be startable by the repository `start-llama.sh` script (`/home/rgardler/projects/llm/start-llama.sh`). To make a new local model available via that script you must add a case block for the model name to the script.
- Steps to add your model to `start-llama.sh`:
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

Notes:
- The proxy passes the configured `llama_model` value through to the local start script. Keep the name consistent between `proxy/config.yaml` `llama_model:` and the `case` pattern in `start-llama.sh`.
- The script lowercases the first argument before matching; use lowercase in your `case` pattern to avoid mismatches.

## Runtime & deployment notes

- Document how to run the local llama-server with the model (example docker run or launch command), expected memory footprint, and any packaging notes.
  - Example using system podman and a self-contained image is documented in `proxy/CONTAINERS.md`.

Example (local launch):

```bash
# Example: run llama-server with the mxbai-embed-large-v1 model on port 8080
docker run --rm -p 8080:8080 --name llama-embed -v /models/mxbai:/models mxbai/llama-server:latest \
  --model /models/mxbai-embed-large-v1
```

## Troubleshooting

- If the proxy returns 404 for a model: check the model config and alias mapping.
- If the embeddings response is empty or the wrong dimension: verify the llama-server exposes the expected embedding dimension and that the model supports embeddings.

## Worked example (mxbai-embed-large-v1)

- Add the example config entry above to the `models:` section of `proxy/config.yaml`.
- Add alias mapping code in `get_model_config()`.
- Add the integration test `proxy/tests/test_embeddings_integration.py`.

## References
- Parent work item: LP-0MN557XBD0H8B8PC - Add an embeddings specific model
