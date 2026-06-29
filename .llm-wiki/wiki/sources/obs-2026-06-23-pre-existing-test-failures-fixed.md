---
type: source
title: "Observation: Pre-existing test failures fixed"
slug: obs-2026-06-23-pre-existing-test-failures-fixed
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: medium
observed_at: 2026-06-23T12:09:12.699Z
tags: ["test-failure", "triage"]
source_context: "Implementing LP-0MQPUCOFJ000QIRF"
---
# 🔍 Observation: Pre-existing test failures fixed
Two pre-existing test failures in test_incremental_ingestion.py were uncovered during LP-0MQPUCOFJ000QIRF implementation. The tests expected _extract_assistant_content and _extract_assistant_content_from_sse to return None for reasoning_content-only responses, but the code now promotes reasoning text as a fallback. Tests updated to expect the promoted content. Child work items LP-0MQQKRLL0000E3I8 and LP-0MQQKS0320004W7S created via triage and then closed after fix.
*Relevance: medium*

*Context: Implementing LP-0MQPUCOFJ000QIRF*

*Tags: test-failure triage*
---
*Observed: 2026-06-23T12:09:12.699Z*