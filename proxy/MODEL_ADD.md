
## How to add a new model to the proxy

This document describes the checklist and a worked example for adding a new model to the proxy. It is intended for engineers who need to add small local or remote models (including embeddings models) to the proxy.

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
