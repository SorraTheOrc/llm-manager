LLM Manager
===========

This repository contains a local copy of the llm proxy with an added model statistics panel and SSE status broadcasts. It was pushed to `SorraTheOrc/llm-manager` for development and review.

Key files
- `proxy/server.py` — server, SSE, and the model stats query helper
- `proxy/tests/` — unit tests and Playwright tests (e.g. `test_query_llama_status.py`, `status-stats.spec.js`)

Quick start (development)
1. Create and activate a virtualenv:
   ```
   python3 -m venv .venv
   . .venv/bin/activate
   ```
2. Install Python test/runtime deps:
   ```
   pip install -r requirements.txt || pip install pytest pytest-asyncio httpx fastapi requests pyyaml
   ```
3. Run the proxy server (from repo root):
   ```
   cd proxy
   . .venv/bin/activate && python -m uvicorn server:app --host 127.0.0.1 --port 3000
   ```

Testing
- Unit tests (example):
  ```
  . .venv/bin/activate && python -m pytest proxy/tests/test_query_llama_status.py -q
  ```
- Full test suite (proxy):
  ```
  . .venv/bin/activate && python -m pytest -q
  ```
- Playwright UI tests (requires Node/npm and browser install):
  ```
  npm i -D playwright && npx playwright install
  npx playwright test proxy/tests/status-stats.spec.js
  ```

Worklog / tracking
- Associated worklog item: `LP-0MN5AW8DE1KJVAKX` (Add stats to the main page for the currently loaded model)

Notes
- The repository was pushed to `git@github.com:SorraTheOrc/llm-manager.git` and `origin` is configured to point to it.
- Before running integration or Playwright tests ensure the local llama-server (or a compatible mock) is available and any long model loading has completed — requests may return 503 while the model loads.

Configuration
-------------

Key server configuration in `proxy/config.yaml` under `server:`:

| Key | Default | Description |
|-----|---------|-------------|
| `slot_management.slot_pool_size` | `4` | Number of slots (GPU contexts) for job-level slot ownership. |
| `slot_management.slot_queue_max_depth` | `16` | Maximum jobs waiting in queue when all slots busy. |
| `slot_management.slot_job_timeout_seconds` | `300.0` | Seconds of inactivity before releasing a job's slot. The timeout check skips slots with an active request in flight to prevent premature slot release during long streaming responses. |
| `slot_management.slot_queue_overflow_retry_after` | `900` | Seconds in Retry-After header on queue overflow. |

When `slot_management` is present in config, the JobScheduler assigns each
multi-turn conversation (session) to a slot for its entire lifetime,
eliminating save/restore overhead between turns. When absent, the previous
hash-based slot assignment with save/restore is used.

Existing session slot settings (used when `slot_management` is absent):

| Key | Default | Description |
|-----|---------|-------------|
| `session_slot_pool_size` | `1` | Number of slots for hash-based assignment. |
| `session_slot_save_path` | `./slot-cache` | Directory for KV cache snapshots. |
| `session_slot_timeout_seconds` | `3.0` | Slot save/restore timeout in seconds. |

Contributing
- Open issues and PRs in the `SorraTheOrc/llm-manager` repo. If you want changes merged upstream to `rgardler/llm`, open a PR from this repo to the upstream repository.

License
- See upstream project for license information.
