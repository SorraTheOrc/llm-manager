"""Lazy tokenizer registry for native (non-tiktoken) token counting.

LP-0MSEQ71IF0003FRT: tiktoken's cl100k encoding undercounts Qwen3-native
tokens ~1.69x for dense prose (60K fixture: tiktoken est 47680 vs actual
~77060), which skewed routing clamps and the persistence cap. This registry
loads vendored, model-specific ``tokenizer.json`` files via the lightweight
``tokenizers`` library so routing/persistence estimates match true token
counts exactly.

Design:
- ``get_tokenizer(name)`` is the single entry point. It NEVER raises —
  unknown names, missing files, or import failures return ``None`` so
  callers can fall back to tiktoken + multiplier.
- Tokenizers load lazily on first use (no startup cost) and are cached.
- New models = one config line (``tokenizer: <name>`` on the model entry)
  + one registry entry mapping the name to its vendored tokenizer.json.
"""

import logging
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

# name -> vendored tokenizer.json path (relative to this module)
TOKENIZER_REGISTRY: dict[str, Path] = {
    "qwen3": Path(__file__).resolve().parent / "tokenizer_data" / "qwen3" / "tokenizer.json",
}


@cache
def _load_tokenizer(name: str):
    """Load and cache the tokenizer for *name*. Raises on failure; callers
    (``get_tokenizer``) translate exceptions into None + warning."""
    from tokenizers import Tokenizer

    path = TOKENIZER_REGISTRY[name]
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer file missing: {path}")
    return Tokenizer.from_file(str(path))


def get_tokenizer(name: str):
    """Return the named tokenizer, or ``None`` when unavailable.

    Never raises into the request path: an unknown name, missing vendored
    file, or ``tokenizers`` import/load error logs a warning and returns
    ``None`` so the caller can fall back to tiktoken + multiplier.
    """
    if not isinstance(name, str) or name not in TOKENIZER_REGISTRY:
        logger.warning("tokenizer %r not in registry (known: %s)", name, sorted(TOKENIZER_REGISTRY))
        return None
    try:
        return _load_tokenizer(name)
    except Exception:
        logger.warning("tokenizer %r failed to load; falling back to tiktoken", name, exc_info=True)
        return None
