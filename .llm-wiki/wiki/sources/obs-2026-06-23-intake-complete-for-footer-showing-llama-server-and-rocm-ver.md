---
type: source
title: "Observation: Intake complete for footer showing llama-server and ROCm versions (LP-0MQLNBRUL002A5DR)"
slug: obs-2026-06-23-intake-complete-for-footer-showing-llama-server-and-rocm-ver
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: medium
observed_at: 2026-06-23T10:42:16.817Z
tags: ["intake", "web-ui", "version", "footer", "rocm", "llama-server"]
source_context: "Intake process for LP-0MQLNBRUL002A5DR"
---
# 🔍 Observation: Intake complete for footer showing llama-server and ROCm versions (LP-0MQLNBRUL002A5DR)
Intake completed for work item LP-0MQLNBRUL002A5DR (Add footer to web UI showing llama-server and ROCm versions). Key decisions: (1) llama-server version via `llama-server --version` at proxy startup; (2) ROCm version via `rocm-smi --showtag` at startup; (3) footer on both index.html and view_logs.html pages; (4) static capture (once at startup, not SSE-polled). Effort: Small (~4.33h). Risk: Low (4/20). Stage: intake_complete.
*Relevance: medium*

*Context: Intake process for LP-0MQLNBRUL002A5DR*

*Tags: intake web-ui version footer rocm llama-server*
---
*Observed: 2026-06-23T10:42:16.817Z*