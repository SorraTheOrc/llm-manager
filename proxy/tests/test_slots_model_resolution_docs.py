"""Documentation contracts for the ``/slots`` model-resolution rule.

Parent: LP-0MVP7HV7Q001AJF8 (router ``/slots`` HTTP 400s). The rule itself is
implemented by LP-0MVU7ZZIA0092AXW; this slice (LP-0MVU7ZZTH005FSFL) documents
it, so the docs *are* the deliverable.

Assertions are behavioural where a behaviour can be exercised and
content-based only where the contract **is** the prose:

* the resolution order ``proxy/README.md`` documents is the order
  :func:`proxy.router_helpers.resolve_local_model_id` actually applies — the
  expected sequence is derived by driving the real resolver through its
  documented precedence, never re-implemented here, so a reordering of the
  implementation fails until the docs follow;
* the documented last-resort id is
  :data:`proxy.router_helpers.LAST_RESORT_MODEL_ID`, so a renamed constant
  cannot leave the prose stale;
* both documents name the three router 400 classes, the guard test that
  fences the *missing* class, and the deferred per-query attribution item
  LP-0MVTDMXXH005D5D5.

Pattern: ``test_contention_queue_docs.py`` / ``test_context_limit_docs.py``
(documentation-drift pins).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from proxy import router_helpers as rh

PROXY_DIR = Path(__file__).resolve().parents[1]
README = PROXY_DIR / "README.md"
TRIAGE_DOC = PROXY_DIR.parent / "docs" / "dev" / "slots-500-triage.md"
GUARD_TEST = PROXY_DIR / "tests" / "test_slots_model_param_guard.py"
DEFERRED_ATTRIBUTION = "LP-0MVTDMXXH005D5D5"
ROUTER_PORT = 8080


def _section(path: Path, heading: str) -> str:
    """Return the body of *heading* up to the next heading of any level."""
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    body = text[start + len(heading):]
    end = re.search(r"\n#{2,4} ", body)
    return body[: end.start()] if end else body


def _table_rows(section: str) -> list[list[str]]:
    """Data rows (header and separator dropped) of the section's markdown table."""
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.strip().startswith("|")
    ]
    return [row for row in rows if not set(row[0]) <= {"-", " "}][1:]


def _row_naming(rows: list[list[str]], keyword: str) -> list[str]:
    return [row for row in rows if keyword in row[0]]


def _models_payload(*ids: str) -> dict:
    return {
        "object": "list",
        "data": [{"id": model_id, "status": {"value": "loaded"}} for model_id in ids],
    }


def _srv(tmp_path, *, pid, current_model=None, models=(), log=None):
    """Minimal router-mode handle with a stubbed ``/models`` fetch."""
    srv = SimpleNamespace(
        log_dir=Path(tmp_path),
        llama_process=SimpleNamespace(pid=pid),
        current_model=current_model,
        config={
            "server": {
                "llama_server_port": ROUTER_PORT,
                "llama_router_mode": True,
            }
        },
        logger=SimpleNamespace(
            warning=lambda *a, **k: None, debug=lambda *a, **k: None
        ),
    )

    async def _router_list_models():
        return _models_payload(*models)

    srv.router_list_models = _router_list_models
    if log is not None:
        (Path(tmp_path) / "llama-server.log").write_text(log)
    return srv


@pytest.fixture(autouse=True)
def _clear_resolution_caches():
    """Reset the resolver's module-level caches around each test."""
    caches = (
        rh._child_port_cache,
        rh._child_port_cache_at,
        rh._child_port_miss_cache,
        rh._child_model_signal_cache,
        rh._router_model_ids_snapshot,
        rh._last_model_resolution,
    )
    for cache in caches:
        cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_model_last_resort_warn_ts = 0.0
    yield
    for cache in caches:
        cache.clear()
    rh._last_child_port_fallback_warn_ts = 0.0
    rh._last_model_last_resort_warn_ts = 0.0


def _observed_sources(tmp_path) -> list[str]:
    """Drive the real resolver down its precedence and collect each ``source``.

    One signal is removed per scenario, so the observed sequence is exactly the
    implemented order: the resolver cannot report ``log_signal`` while a
    higher-priority signal is still available without reordering its own code.
    """
    scopes = []
    for index, (kwargs, explicit) in enumerate(
        (
            (
                dict(
                    current_model="Qwen2.5-7B",
                    models=("Qwen3-32B", "mxbai-embed"),
                    log="srv  proxy_reques: proxying request to model Qwen3 on port 41395\n",
                ),
                "mxbai-embed",
            ),
            (
                dict(
                    current_model="Qwen2.5-7B",
                    models=("Qwen3-32B", "mxbai-embed"),
                    log="srv  proxy_reques: proxying request to model Qwen3 on port 41395\n",
                ),
                None,
            ),
            (dict(current_model=None, models=("Solo-7B", "mxbai-embed")), None),
            (
                dict(
                    current_model=None,
                    models=("Qwen3-32B", "Qwen2.5-7B"),
                    log="srv  proxy_reques: proxying request to model Qwen3 on port 41395\n",
                ),
                None,
            ),
            (
                dict(
                    current_model=None,
                    models=(),
                    log="srv  proxy_reques: proxying request to model Qwen3-Coder on port 41395\n",
                ),
                None,
            ),
            (dict(current_model=None, models=()), None),
        )
    ):
        scope = tmp_path / f"scope-{index}"
        scope.mkdir()
        srv = _srv(scope, pid=40000 + index, **kwargs)
        asyncio.run(rh.ensure_router_model_ids(srv))
        scopes.append(rh.resolve_local_model_id(srv, explicit))
    return [source for _model, source in scopes]


class TestReadmeResolutionRule:
    def test_documented_order_matches_the_implemented_order(self, tmp_path):
        """README lists the resolver's sources in the order it actually applies."""
        section = _section(README, "#### Model resolution rule")
        observed = _observed_sources(tmp_path)

        assert len(set(observed)) == len(observed) == 6, observed
        positions = [section.index(f"`source={source}`") for source in observed]
        assert positions == sorted(positions), (
            "proxy/README.md documents the model-resolution sources out of the "
            f"order resolve_local_model_id applies: observed {observed}"
        )

    def test_documented_last_resort_is_the_shipped_constant(self):
        """The prose names the one id the implementation uses as last resort."""
        section = _section(README, "#### Model resolution rule")
        assert f'"{rh.LAST_RESORT_MODEL_ID}"' in section
        assert "last resort" in section

    def test_documented_rule_targets_the_discovered_child_port(self):
        """The resolution subsection states the port/target consequence."""
        section = _section(README, "#### Model resolution rule")
        assert "discovered child port" in section
        assert "?model=" in section

    def test_readme_distinguishes_the_three_400_classes(self):
        """The README table carries missing / unknown / unloaded semantics."""
        rows = _table_rows(_section(README, "#### Model resolution rule"))
        missing = _row_naming(rows, "Missing")
        unknown = _row_naming(rows, "Unknown")
        unloaded = _row_naming(rows, "Unloaded")

        assert len(missing) == len(unknown) == len(unloaded) == 1, rows
        assert "Proxy defect" in missing[0][2]
        assert "Proxy defect" in unknown[0][2]
        assert "Expected truth" in unloaded[0][2]
        assert "slots_failure_is_definitive_model_absent" in unloaded[0][3]

    def test_readme_names_the_guard_test_and_the_deferred_item(self):
        section = _section(README, "#### Model resolution rule")
        assert "test_slots_model_param_guard.py" in section
        assert "test_slots_model_id_resolution.py" in section
        assert DEFERRED_ATTRIBUTION in section
        assert GUARD_TEST.is_file()


class TestTriageDoc400Semantics:
    def test_triage_doc_has_a_400_semantics_section(self):
        section = _section(
            TRIAGE_DOC, "## The three router `/slots` 400 classes"
        )
        assert section.strip()

    def test_three_classes_are_distinct_rows(self):
        """Each class states its own meaning and its own enforcement/response."""
        rows = _table_rows(
            _section(TRIAGE_DOC, "## The three router `/slots` 400 classes")
        )
        missing = _row_naming(rows, "Missing")
        unknown = _row_naming(rows, "Unknown")
        unloaded = _row_naming(rows, "Unloaded")

        assert len(missing) == len(unknown) == len(unloaded) == 1, rows
        # The *missing* class is a proxy defect fenced by the guard test.
        assert "Proxy defect" in missing[0][2]
        assert "test_slots_model_param_guard.py" in missing[0][3]
        # The *unknown* class is a resolution defect, not a router quirk.
        assert "Proxy defect" in unknown[0][2]
        assert "resolve_local_model_id" in unknown[0][3]
        # The *unloaded* class is expected truth the proxy fail-opens on.
        assert "Expected truth" in unloaded[0][2]
        assert "slots_failure_is_definitive_model_absent" in unloaded[0][3]

    def test_triage_doc_names_the_deferred_attribution_item(self):
        section = _section(
            TRIAGE_DOC, "## The three router `/slots` 400 classes"
        )
        assert DEFERRED_ATTRIBUTION in section
