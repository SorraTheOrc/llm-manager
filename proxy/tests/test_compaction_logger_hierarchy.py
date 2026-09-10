
# <!-- REFACTOR-LP-0MTV87EC4006EU5Z
# smell: unused_import
# severity: critical
# description: Local variable `fields` is assigned to but never used
# -->
"""
Regression: compaction dry-run advisory logging suppressed by logger hierarchy.

LP-0MTV5GFL8007WSK2 — AC1/AC2/AC4.

Root cause: proxy/proxy/compaction.py et al used getLogger(__name__)
→ proxy.compaction which parents to root (WARNING), not llama-proxy (INFO).
log_compaction_event() emits tidy compacts at INFO, so isEnabledFor(INFO)
was False and 0 advisory lines reached proxy.log.

Fix: loggers renamed to llama-proxy.* hierarchy so INFO propagates via
llama-proxy's TimedRotatingFileHandler (setup_logging in proxy/utils.py).

This file guards the fix against regression: INFO compaction_event lines
must be observable via the llama-proxy hierarchy (matching production
setup_logging), and WARNING paths must remain WARNING.
"""

import logging

import proxy.compaction as comp_mod
import proxy.compaction_summarizer as summ_mod
import proxy.session as sess_mod
import proxy.tokenizers as tok_mod
import pytest
from proxy.compaction import log_compaction_event, plan_session_compaction

_EST_PER_MSG = 1000


def _est(messages) -> int:
    return _EST_PER_MSG * len(messages) + sum(len(str(m.get("content", ""))) for m in messages)


def _fast_config():
    return {"server": {"local_model_ctx_size": 262144, "session_slot_pool_size": 3, "compaction_trigger_ratio": 0.70}}


def _summ(middle, prev=None):
    return f"SUMMARY {len(middle)} msgs"


def _session(n: int):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "FIRST"}, {"role": "assistant", "content": "REPLY"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


class TestLoggerHierarchy:
    def test_compaction_logger_is_llama_proxy_child(self):
        assert comp_mod.logger.name == "llama-proxy.compaction"
        assert comp_mod.logger.name.startswith("llama-proxy.")

    def test_summarizer_logger_is_llama_proxy_child(self):
        assert summ_mod.logger.name == "llama-proxy.compaction_summarizer"

    def test_session_logger_is_llama_proxy_child(self):
        assert sess_mod.logger.name == "llama-proxy.session"

    def test_tokenizers_logger_is_llama_proxy_child(self):
        assert tok_mod.logger.name == "llama-proxy.tokenizers"

    def test_compaction_info_enabled_when_llama_proxy_info(self, caplog):
        # Simulate production setup_logging: only llama-proxy at INFO.
        # Then INFO on compaction child must be enabled (parents to llama-proxy, not root).
        caplog.set_level(logging.INFO, logger="llama-proxy")
        # Ensure root stays WARNING (default) so proxy.* would fail.
        assert logging.getLogger("llama-proxy.compaction").isEnabledFor(logging.INFO)
        assert logging.getLogger("llama-proxy.compaction_summarizer").isEnabledFor(logging.INFO)

    def test_dry_run_compact_emits_at_info_via_llama_proxy_hierarchy(self, caplog):
        result = plan_session_compaction(_session(60), _fast_config(), "fast", summarizer=_summ, estimate_tokens=_est)
        assert result["action"] == "compact"
        with caplog.at_level(logging.INFO, logger="llama-proxy.compaction"):
            fields = log_compaction_event(result, session_id="sess-abc12345", dry_run=True, estimate_tokens=_est)
        assert fields is not None
        assert fields["dry_run"] is True
        recs = [r for r in caplog.records if r.name == "llama-proxy.compaction" and "compaction_event" in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.INFO
        assert "dry_run=True" in recs[0].getMessage()
        assert "session=sess-abc" in recs[0].getMessage()

    def test_non_dry_run_compact_emits_at_info(self, caplog):
        result = plan_session_compaction(_session(60), _fast_config(), "fast", summarizer=_summ, estimate_tokens=_est)
        with caplog.at_level(logging.INFO, logger="llama-proxy.compaction"):
            fields = log_compaction_event(result, session_id="sess-xyz", dry_run=False, estimate_tokens=_est)
        rec = next(r for r in caplog.records if r.name == "llama-proxy.compaction")
        assert rec.levelno == logging.INFO
        assert "dry_run=False" in rec.getMessage()

    def test_warning_paths_remain_warning(self, caplog):
        # backstop / over-budget path → WARNING even after hierarchy fix (AC4)
        def big_summ(middle, prev=None):
            return "B" * 20000

        result = plan_session_compaction(_session(60), _fast_config(), "fast", summarizer=big_summ, estimate_tokens=_est, backstop=True)
        # either backstop_dropped or exhausted → WARNING
        with caplog.at_level(logging.INFO, logger="llama-proxy.compaction"):
            log_compaction_event(result, session_id="sess-warn", dry_run=True, estimate_tokens=_est)
        rec = next(r for r in caplog.records if r.name == "llama-proxy.compaction")
        assert rec.levelno == logging.WARNING

    def test_noop_produces_no_event_even_at_info(self, caplog):
        result = plan_session_compaction(_session(2), _fast_config(), "fast", summarizer=_summ, estimate_tokens=_est)
        assert result["action"] == "noop"
        with caplog.at_level(logging.INFO, logger="llama-proxy.compaction"):
            fields = log_compaction_event(result, session_id="sess-noop", dry_run=True, estimate_tokens=_est)
        assert fields is None
        assert not [r for r in caplog.records if r.name == "llama-proxy.compaction"]
