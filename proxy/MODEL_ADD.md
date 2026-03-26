
## How to add a new model to the proxy

This document describes the checklist and a worked example for adding a new model to the proxy. It is intended for engineers who need to add small local or remote models (including embeddings models) to the proxy.

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

## Model configuration

 - Add an entry under the `models:` section of `proxy/config.yaml`. Required/typical fields (follow existing config style):
   - `type`: `local` or `remote` (matches current `proxy/config.yaml` schema)
   - `llama_model`: for `type: local` entries, the model name used by the llama start script (e.g. `mxbai-embed-large-v1`)
   - `endpoint` / `api_key_env`: for `type: remote` entries, the remote endpoint and API key env var
  - `aliases`: optional list of aliases (e.g. `embeddings`)

Example (add under `models:` in `proxy/config.yaml`, matching the existing schema in `proxy/config.yaml`):

```yaml
models:
  mxbai-embed-large-v1:
    type: "local"
    llama_model: "mxbai-embed-large-v1"
    aliases:
      - "embeddings"
    # Note: do not include a `supports` field here — the proxy will forward requests and the backend
    # (llama-server or remote provider) will surface errors if a model does not support a capability.

  # Remote example
  example-remote:
    type: "remote"
    endpoint: "https://models.example.com"
    api_key_env: "EXAMPLE_API_KEY"
    aliases:
      - "example-*"
```

## Alias resolution

- The proxy should resolve aliases before model lookup. Add alias handling where model names are resolved (e.g. in `get_model_config()`):
  - If `model` matches an alias, replace with the canonical `name`.

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
- Parent work item: LP-0MN557XBD0H8B8PC — Add an embeddings specific model
