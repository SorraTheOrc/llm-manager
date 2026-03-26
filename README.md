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

Contributing
- Open issues and PRs in the `SorraTheOrc/llm-manager` repo. If you want changes merged upstream to `rgardler/llm`, open a PR from this repo to the upstream repository.

License
- See upstream project for license information.
