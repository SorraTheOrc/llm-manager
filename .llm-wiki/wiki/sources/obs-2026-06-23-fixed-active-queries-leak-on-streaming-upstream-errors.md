---
type: source
title: "Observation: Fixed active_queries leak on streaming upstream errors"
slug: obs-2026-06-23-fixed-active-queries-leak-on-streaming-upstream-errors
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: high
observed_at: 2026-06-23T12:09:12.699Z
tags: ["bug", "proxy", "active_queries"]
source_context: "Implementing LP-0MQPUCOFJ000QIRF"
---
# ⭐ Observation: Fixed active_queries leak on streaming upstream errors
Bug LP-0MQPUCOFJ000QIRF: Streaming upstream errors (429/5xx) leaked the active_queries counter in proxy/proxy/router.py. The buffered error path (lines ~335-368) returned Response() without calling _decrement_active_queries(srv). Fix: added await _decrement_active_queries(srv) at line 361 before the return. Three new tests verify 429/500 decrement and concurrent request isolation. Commit 80985e1 pushed to dev.
*Relevance: high*

*Context: Implementing LP-0MQPUCOFJ000QIRF*

*Tags: bug proxy active_queries*
---
*Observed: 2026-06-23T12:09:12.699Z*