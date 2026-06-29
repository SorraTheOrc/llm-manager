---
type: source
title: "Observation: Cache fill progress display regression root cause"
slug: obs-2026-06-23-cache-fill-progress-display-regression-root-cause
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: high
observed_at: 2026-06-23T17:44:46.919Z
tags: ["proxy", "regression", "cache-fill", "progress", "stream-output"]
source_context: "Intake process for LP-0MQPU0BKC008TNGC (Proxy log is no longer showing cache fill progress)"
---
# ⭐ Observation: Cache fill progress display regression root cause
The proxy console no longer shows cache fill/prefill progress from llama-server. Root cause: the `_stream_output` closure in `lifecycle.py:start_llama_server()` had its prompt processing progress display code (~35 lines) inadvertently removed during the host-first startup refactor in commit `6012013` (LP-0MQNZXZUH002FTON). The `extract_progress_data()` and `format_progress()` helper functions in `handlers.py` still exist and are usable. Fix is to restore the progress display code in `_stream_output` in `proxy/proxy/lifecycle.py`.
*Relevance: high*

*Context: Intake process for LP-0MQPU0BKC008TNGC (Proxy log is no longer showing cache fill progress)*

*Tags: proxy regression cache-fill progress stream-output*
---
*Observed: 2026-06-23T17:44:46.919Z*