"""Fallback-line parsing for the munitoai budget benchmark.

``proxy/benchmarks/munitoai_budget_eval.py`` attributes munitoai-free spills by
parsing ``Fallback triggered ... from=munitoai-free, to=..., reason=...`` lines.
LP-0MVMBMSWR0036RES appended ``session=``/``attempt=`` (and budget fields for
``rate_limited*``) after ``reason=``, so the extractor must stay bounded to the
reason rather than to end-of-line.
"""

from __future__ import annotations

from datetime import datetime

import pytest

bench = pytest.importorskip("proxy.benchmarks.munitoai_budget_eval")

TS = datetime(2026, 10, 23, 9, 14, 2)


def test_legacy_session_less_line_still_parses():
    line = (
        "2026-10-23 09:14:02,884 - INFO - Fallback triggered for model=v1/chat/completions, "
        "from=munitoai-free, to=opencode-go, reason=rate_limited"
    )
    ev = bench.parse_event(TS, line)
    assert ev is not None
    assert ev["kind"] == "fallback"
    assert ev["to"] == "opencode-go"
    assert ev["reason"] == "rate_limited"


def test_session_bearing_line_parses_with_budget_evidence():
    line = (
        "2026-10-23 09:14:02,884 - INFO - Fallback triggered for model=v1/chat/completions, "
        "from=munitoai-free, to=opencode-go, reason=rate_limited "
        "session=herdr-1792767061-1282111-17770 attempt=4 limit=60 remaining=0 waited=12.500"
    )
    ev = bench.parse_event(TS, line)
    assert ev is not None
    assert ev["kind"] == "fallback"
    assert ev["to"] == "opencode-go"
    assert ev["reason"] == "rate_limited"


def test_reason_with_spaces_is_not_truncated():
    line = (
        "2026-10-23 09:14:02,884 - INFO - Fallback triggered for model=v1/chat/completions, "
        "from=munitoai-free, to=opencode-go, reason=HTTP 429 "
        "session=unknown attempt=1"
    )
    ev = bench.parse_event(TS, line)
    assert ev is not None
    assert ev["reason"] == "HTTP 429"
