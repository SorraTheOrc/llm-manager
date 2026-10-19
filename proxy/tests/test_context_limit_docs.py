"""Documentation contracts for the per-entry ``max_context_tokens`` gate (F4).

Parent: LP-0MVF2DNCX003Z5CO (per-model context size). Implementation:
LP-0MVF9N2QL001CDA7 (gate), LP-0MVF9N31A0006VH5 (shipped values),
this item LP-0MVF9N3BV00155RI (docs + observability).

The docs are the deliverable here, so the assertions are behavioural where a
behaviour can be exercised and content-based only where the contract *is* the
prose:

* every fenced ``yaml`` example in the three touched documents parses, and the
  ``max_context_tokens`` examples load through the **production** merged-config
  path (``proxy.utils.load_merged_config``), then feed the resolved entries to
  the real gate helpers — so a copy-pasted example is provably accepted by the
  shipped schema rather than merely syntactically valid;
* each document states the field name, type, disabled semantics, the
  local-entry composition rule, the skip-vs-failure distinction and the
  backend-signal counter.
"""

import re
from pathlib import Path

import pytest
import yaml
from proxy.provider import (
    _provider_context_limit,
    _should_skip_provider_context,
)
from proxy.utils import load_merged_config

PROXY_DIR = Path(__file__).resolve().parents[1]

DOCS = {
    "MODEL_ADD.md": PROXY_DIR / "MODEL_ADD.md",
    "README.md": PROXY_DIR / "README.md",
    "routing.md": PROXY_DIR / "docs" / "routing.md",
}

#: Documents that must carry a copy-pasteable ``max_context_tokens`` example.
EXAMPLE_DOCS = ("MODEL_ADD.md", "README.md")

_YAML_FENCE = re.compile(r"```yaml\n(.*?)```", re.S)


def _yaml_blocks(path: Path) -> list[str]:
    return _YAML_FENCE.findall(path.read_text())


def _context_examples(path: Path) -> list[str]:
    return [block for block in _yaml_blocks(path) if "max_context_tokens" in block]


@pytest.mark.parametrize("doc_name", DOCS, ids=list(DOCS))
def test_every_yaml_example_parses(doc_name):
    """AC: doc examples are copy-pasteable YAML (they must at least parse)."""
    blocks = _yaml_blocks(DOCS[doc_name])
    assert blocks, f"{doc_name} has no fenced YAML examples"
    for index, block in enumerate(blocks):
        try:
            yaml.safe_load(block)
        except yaml.YAMLError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{doc_name} yaml block #{index} does not parse: {exc}")


@pytest.mark.parametrize("doc_name", DOCS, ids=list(DOCS))
def test_docs_state_the_field_type(doc_name):
    """AC: the type (``int``) is stated alongside the field name."""
    text = re.sub(r"\s+", " ", DOCS[doc_name].read_text())
    assert re.search(r"`max_context_tokens`[^.\n]{0,40}\bint\b", text), (
        f"{doc_name} does not state that max_context_tokens is an int"
    )


@pytest.mark.parametrize("doc_name", EXAMPLE_DOCS)
def test_context_limit_example_loads_through_the_shipped_config_path(doc_name, tmp_path):
    """AC: the ``max_context_tokens`` example parses against the shipped schema.

    The example is written as an overlay and resolved with the same production
    loader the proxy uses (base ``config.yaml`` deep-merged, then validated),
    so the resolved entries are the ones the gate would actually see.
    """
    examples = _context_examples(DOCS[doc_name])
    assert examples, f"{doc_name} has no max_context_tokens YAML example"

    for index, block in enumerate(examples):
        overlay = tmp_path / f"{doc_name}-{index}.yaml"
        overlay.write_text(block)
        config = load_merged_config(overlay)

        entries = [
            entry
            for model in (config.get("models") or {}).values()
            if isinstance(model, dict)
            for entry in (model.get("providers") or [])
            if isinstance(entry, dict)
        ]
        gated = [entry for entry in entries if "max_context_tokens" in entry]
        ungated = [entry for entry in entries if "max_context_tokens" not in entry]
        assert gated, f"{doc_name} example #{index} declares no capped entry"
        assert ungated, (
            f"{doc_name} example #{index} must show an ungated destination so "
            "the disabled semantics are copy-pasteable"
        )

        for entry in gated:
            raw = entry["max_context_tokens"]
            assert isinstance(raw, int) and not isinstance(raw, bool)
            limit = _provider_context_limit(entry)
            assert limit == raw > 0
            # Inclusive ceiling: at the limit dispatches, one above skips.
            assert _should_skip_provider_context(entry, limit) == (False, limit)
            assert _should_skip_provider_context(entry, limit + 1) == (True, limit)

        # An entry without the field is ungated at any size (disabled semantics).
        for entry in ungated:
            assert _provider_context_limit(entry) == 0
            assert _should_skip_provider_context(entry, 10_000_000) == (False, 0)

        # A local entry in the example must be gated too (the docs' composition
        # rule applies the field to local entries as an extra hard cap).
        local = [entry for entry in gated if entry.get("type") == "local"]
        assert local, f"{doc_name} example #{index} must show the local composition"


#: Semantics each document must state for the field (AC: field name + type,
#: disabled semantics, local composition rule, skip-vs-failure distinction and
#: the backend-signal counter).
REQUIRED_SEMANTICS = {
    "field name": "max_context_tokens",
    "disabled semantics": "non-positive",
    "local composition rule": "min(existing local thresholds, max_context_tokens)",
    "skip status": "provider_context_exceeded",
    "no cooldown on skip": "cooldown",
    "no affinity on skip": "affinity",
    "no contention-queue entry on skip": "contention-queue",
    "entry stays eligible": "eligible for later",
    "backend signal counter": "backend_signals.provider_context_exceeded",
}


@pytest.mark.parametrize("doc_name", DOCS, ids=list(DOCS))
def test_docs_state_the_field_semantics(doc_name):
    """AC: every touched document carries the agreed field semantics."""
    # Normalise whitespace so a phrase wrapped across Markdown lines still
    # counts as documented (the prose is the contract, not the line breaks).
    text = re.sub(r"\s+", " ", DOCS[doc_name].read_text())
    missing = [needle for needle in REQUIRED_SEMANTICS.values() if needle not in text]
    assert not missing, f"{doc_name} is missing documented semantics: {missing}"


@pytest.mark.parametrize("doc_name", ("MODEL_ADD.md", "README.md"))
def test_provider_entry_tables_carry_the_field_type(doc_name):
    """AC: the provider-entry field tables list the field with its type."""
    text = DOCS[doc_name].read_text()
    assert re.search(r"\|\s*`max_context_tokens`\s*\|\s*int\s*\|", text), (
        f"{doc_name} provider-entry table has no int-typed max_context_tokens row"
    )
