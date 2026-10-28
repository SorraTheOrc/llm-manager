#!/usr/bin/env python3
"""KV slot snapshot save/restore benchmark harness (LP-0MVOG8SYO006R49J).

Measures, per prompt-size band (10k / 25k / 40k / 60k+), what a llama-server
KV *slot snapshot* costs: the wall time of ``POST /slots/{id}?action=save``
and ``action=restore``, the on-disk bytes of the snapshot, the HTTP outcome,
and — when a verification completion is enabled — llama-server's
``timings.prompt_n`` so the restored-prefix hit can be computed.

The measurements feed the parent evaluation
(``LP-0MVN3KPEC007WWL5`` AC1) and the virtual-slot spike
(``LP-0MVGD9ZKI004ETTE``), which both need save/restore cost against the live
adaptive-timeout model rather than an estimate.

Hit definition (token based, latency is corroboration only)
-----------------------------------------------------------
A restore *hit* is defined by tokens, never by wall time::

    cached_tokens = prompt_total - timings.prompt_n

where ``prompt_total`` is the request's total prompt tokens (``usage.
prompt_tokens`` when the verification completion reports it, otherwise the
harness estimate) and ``timings.prompt_n`` is the number of prompt tokens
llama-server actually processed. A restore that took 40 s but still made the
server reprocess 24k of a 25k prompt is a *miss*, however fast; a restore
that took 58 s but left only 500 tokens to process is a *hit*. Latency is
recorded for the timeout comparison only and is never part of the hit test.

Timeout model
-------------
Every save/restore is issued with the *live* adaptive timeout
(``proxy/proxy/session.py:_build_slot_context``,
``proxy/config.yaml`` slot-persistence safeguards)::

    min(session_slot_timeout_seconds + session_slot_timeout_per_token_seconds
        * estimated_tokens, session_slot_max_timeout_seconds)

i.e. ``min(3.0 + 0.0015 * tokens, 60)`` with the shipped values. Each record
carries the timeout it was issued under and whether the operation finished
inside it, because the whole question the parent asks is *where* that model
stops describing reality.

Wall clock vs server time
-------------------------
A wall-clock save/restore is the llama-server request round trip: it includes
any queueing behind an in-flight task on another slot. The record therefore
also carries llama-server's own ``timings.save_ms`` / ``timings.restore_ms``
as ``save_server_ms`` / ``restore_server_ms``, so a timeout can be attributed
to *server work* or to *time spent waiting for the server* instead of being
blamed on the mechanism. ``snapshot_bytes`` is taken from the save response's
``n_written`` (authoritative) with the on-disk stat kept alongside as
``snapshot_bytes_on_disk``; statting alone could read a previous band's file.

Populate, erase and slot pinning
--------------------------------
A band is only measurable if the slot holds *that band's* KV before the save:

* ``--populate-slot`` prefills the band prompt into the measured slot (pinned
  with llama-server's ``id_slot``), so the snapshot is prompt-sized;
* each populated band first issues ``action=erase`` so the previous band's KV
  cannot leak into the next band's snapshot or restore;
* ``--verify-slot-id`` pins the post-restore verification completion to the
  restored slot, which makes ``timings.prompt_n`` describe the restore;
* ``--load-slot-id`` pins the ``--under-load`` stream to a *different* slot,
  so "another slot is processing" is a real condition.

Usage
-----
::

    # Offline check — no llama-server, no proxy, no GPU required
    python3 -m proxy.benchmarks.slot_snapshot_benchmark --dry-run
    python3 -m proxy.benchmarks.slot_snapshot_benchmark --dry-run --json

    # Measure the four bands against a live llama-server (slot 0)
    python3 -m proxy.benchmarks.slot_snapshot_benchmark \\
        --base-url http://localhost:8080 --snapshot-dir /path/to/slot-cache

    # AC1 "under load": keep a second slot processing while saves issue
    python3 -m proxy.benchmarks.slot_snapshot_benchmark --under-load \\
        --bands 10k,25k,40k,60k+ --json

    # Add the token-based hit measurement (one non-streaming completion per
    # band after the restore; see the fidelity caveats in
    # docs/slot-snapshot-benchmark-harness.md)
    python3 -m proxy.benchmarks.slot_snapshot_benchmark --verify-chat --json

Failure policy
--------------
A failed save or restore is a *measurement outcome*, not a harness failure.
It is recorded with its error and its elapsed time, is never retried, and the
run continues (a retry would silently replace a timeout with a success and
destroy the very data AC1 needs). When the save fails the restore is recorded
as ``skipped`` instead of being issued against a snapshot that was never
written.

This module is an analysis tool: no production module may import it
(:data:`SNAPSHOT_BENCHMARK_SENTINEL` guards that, matching the
``prefill_thrash_replay_harness`` AC5 pattern).

Reuse (deliberately not rebuilt): the result-directory convention
(``LLAMA_PROXY_BENCHMARK_ROOT``) and the salted-prompt/concurrency drivers
from ``proxy/benchmarks/slot_benchmark.py``; the per-token KV footprint from
``proxy/benchmarks/kv_memory_table.py`` for the snapshot-size cross-check; the
``slot_save``/``slot_restore`` log vocabulary parsed by
``scripts/slot_persistence_harness.py`` (matchers in
``proxy/proxy/session.py:_call_slot_endpoint``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

try:  # pragma: no cover - httpx absence is a live-run failure, not a test one
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from proxy.benchmarks.kv_memory_table import KV_MIB_PER_TOKEN
except ImportError:  # pragma: no cover - depends on the import root
    try:
        from benchmarks.kv_memory_table import KV_MIB_PER_TOKEN
    except ImportError:  # pragma: no cover - exercised when run as a script
        from kv_memory_table import KV_MIB_PER_TOKEN

#: Copied into any module that imports this harness, so the AC5 guard test can
#: prove the harness stayed out of the production import graph.
SNAPSHOT_BENCHMARK_SENTINEL = "slot-snapshot-save-restore-benchmark"

#: Repo root whose live config the benchmark describes. Matches
#: ``slot_benchmark.PROJECT_ROOT`` (reuse of the result-dir convention):
#: ``LLAMA_PROXY_BENCHMARK_ROOT`` lets the harness code live in a worktree
#: while it operates on the live checkout's config.
PROJECT_ROOT = Path(
    os.environ.get("LLAMA_PROXY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parent.parent.parent
)
CONFIG_YAML = PROJECT_ROOT / "proxy" / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark-results"

MIB = 1024 * 1024

# ---------------------------------------------------------------------------
# Token / prompt sizing
# ---------------------------------------------------------------------------
#: Character heuristic for prompt sizing. The harness emits numeric content
#: (comma-separated integers, like ``slot_benchmark.build_varied_content``), for
#: which the production tokeniser (Qwen3.6-35B-A3B) consumes **~1.0 chars per
#: token**: every comma-separated integer is its own token. Measured on the
#: production host 2026-10-28: a 40 125-character band prompt was 40 062 prompt
#: tokens (``usage.prompt_tokens``), i.e. 1.0016 chars/token. Assuming 4.0
#: over-delivered a nominal "10k" prompt to 40 039 tokens — outside its band and
#: a 4x error against the timeout model. The value is an estimate by design: the
#: true total is recorded from ``usage.prompt_tokens`` whenever ``--verify-chat``
#: or ``--populate-slot`` runs (see the fidelity caveats in the docs).
CHARS_PER_TOKEN = 1.0

#: One digit group emitted per block. Commas delimit every integer, so each
#: integer is one token and the prompt is sized in blocks rather than characters.
NUMBERS_PER_BLOCK = 200

#: Unique leading marker per (salt, band): band prompts must not share a
#: prefix, otherwise the 40k band would restore the 10k band's KV prefix and
#: the bands would stop being independent measurements.
SALT_HEADER_TEMPLATE = "BENCHMARK SNAPSHOT SALT {salt} TARGET {target_tokens}\n"

PROMPT_TAIL = "\nCount the numbers above and reply with the exact total.\n"


@dataclass(frozen=True)
class BandSpec:
    """One prompt-size band: a human name and its token floor."""

    name: str
    target_tokens: int


#: The four bands the parent evaluation asks for (AC1). ``60k+`` is the open
#: band: 60 000 tokens is its floor, not its ceiling — measure longer prompts
#: by passing a larger token count to ``--bands``.
DEFAULT_BANDS: tuple[BandSpec, ...] = (
    BandSpec("10k", 10_000),
    BandSpec("25k", 25_000),
    BandSpec("40k", 40_000),
    BandSpec("60k+", 60_000),
)

_BANDS_BY_NAME = {band.name: band for band in DEFAULT_BANDS}


def _salt_seed(salt: str) -> int:
    """Deterministic RNG seed for *salt* (independent of ``PYTHONHASHSEED``)."""
    digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _number_block(rng: random.Random, count: int = NUMBERS_PER_BLOCK) -> str:
    return ",".join(str(rng.randint(1, 999_999)) for _ in range(count)) + ",\n"


def _block_tokens(block: str) -> int:
    """Estimated tokens in one numeric block (one token per integer).

    The production tokeniser splits ``1,234,567,`` into one token per integer
    plus a punctuation token, so a block of *n* integers with *n* separators is
    ``2n`` tokens. Uses the same 1.0-chars-per-token model as
    :data:`CHARS_PER_TOKEN`, so :func:`estimated_prompt_tokens` stays consistent
    with :func:`build_salted_prompt` for any prompt either of them produces.
    """
    return max(1, math.ceil(len(block) / CHARS_PER_TOKEN))


def estimated_prompt_tokens(text: str) -> int:
    """Estimated token count of *text* (character heuristic, see module docs)."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def build_salted_prompt(target_tokens: int, salt: str) -> str:
    """Deterministic prompt of *target_tokens* (never more) with a unique salt.

    The salt is the *first* thing in the prompt so two bands (or two runs)
    never share a KV prefix: a shared prefix would let the shorter band's
    snapshot serve the longer band's restore and silently understate both.

    Sizing is a token budget, not a character budget: the header and tail are
    counted first and numeric blocks (one token per integer) fill the
    remainder, so the result does not overshoot its band. Overshooting is not
    harmless — it moves a prompt out of the band it is supposed to represent.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    header = SALT_HEADER_TEMPLATE.format(salt=salt, target_tokens=target_tokens)
    rng = random.Random(_salt_seed(salt))
    tokens_used = _block_tokens(header) + _block_tokens(PROMPT_TAIL)
    blocks: list[str] = []
    while tokens_used < target_tokens:
        block = _number_block(rng)
        cost = _block_tokens(block)
        if tokens_used + cost > target_tokens:
            # Trim the block to the remaining budget instead of overshooting:
            # one integer (and its comma) is one token.
            remaining = target_tokens - tokens_used
            block = block[:remaining]
            cost = _block_tokens(block)
            if remaining <= 0:
                break
        blocks.append(block)
        tokens_used += cost
    return header + "".join(blocks) + PROMPT_TAIL


def band_prompt(band: BandSpec, salt: str) -> str:
    """Prompt for *band*, salted with *salt*."""
    return build_salted_prompt(band.target_tokens, salt)


def parse_bands(spec: str | None) -> tuple[BandSpec, ...]:
    """Parse a ``--bands`` value: names (``10k``/``60k+``) and/or token counts."""
    if spec is None or not str(spec).strip():
        return DEFAULT_BANDS
    bands: list[BandSpec] = []
    seen: set[str] = set()
    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue
        if token in _BANDS_BY_NAME:
            band = _BANDS_BY_NAME[token]
        elif token.isdigit():
            band = BandSpec(f"{int(token)}", int(token))
        else:
            raise ValueError(
                f"unknown band {token!r}: use one of "
                f"{sorted(_BANDS_BY_NAME)} or a token count"
            )
        if band.name in seen:
            raise ValueError(f"duplicate band {band.name!r}")
        seen.add(band.name)
        bands.append(band)
    if not bands:
        raise ValueError("--bands selected no bands")
    return tuple(bands)


# ---------------------------------------------------------------------------
# Adaptive timeout model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeoutModel:
    """The live slot-persistence timeout model (three config keys)."""

    base_seconds: float = 3.0
    per_token_seconds: float = 0.0015
    cap_seconds: float = 60.0

    def timeout_for(self, tokens: int) -> float:
        """``min(base + per_token * tokens, cap)`` — the production formula."""
        return min(
            self.base_seconds + self.per_token_seconds * max(0, int(tokens)),
            self.cap_seconds,
        )

    @property
    def cap_tokens(self) -> float:
        """Prompt size at which the timeout reaches its cap."""
        if self.per_token_seconds <= 0:
            return math.inf
        return (self.cap_seconds - self.base_seconds) / self.per_token_seconds


#: The values shipped in ``proxy/config.yaml`` (and the numbers the parent's
#: AC1 quotes): ``min(3.0 + 0.0015 * tokens, 60)``.
DEFAULT_TIMEOUT_MODEL = TimeoutModel()

_TIMEOUT_KEYS = {
    "base_seconds": "session_slot_timeout_seconds",
    "per_token_seconds": "session_slot_timeout_per_token_seconds",
    "cap_seconds": "session_slot_max_timeout_seconds",
}


def timeout_model_from_config(server_config: dict | None) -> TimeoutModel:
    """Build a :class:`TimeoutModel` from a ``server:`` config mapping."""
    config = server_config or {}

    def _read(field_name: str, default: float) -> float:
        value = config.get(_TIMEOUT_KEYS[field_name])
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    return TimeoutModel(
        base_seconds=_read("base_seconds", DEFAULT_TIMEOUT_MODEL.base_seconds),
        per_token_seconds=_read(
            "per_token_seconds", DEFAULT_TIMEOUT_MODEL.per_token_seconds
        ),
        cap_seconds=_read("cap_seconds", DEFAULT_TIMEOUT_MODEL.cap_seconds),
    )


def load_server_config(config_yaml: Path | None = None) -> dict:
    """Load the ``server:`` block of config.yaml, or ``{}`` when unavailable.

    Fail-open: the harness is routinely run from a worktree (or in tests) where
    the live config is absent; the documented shipped defaults then apply.
    """
    path = Path(config_yaml) if config_yaml is not None else CONFIG_YAML
    try:
        import yaml

        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        server = data.get("server") if isinstance(data, dict) else None
        return dict(server) if isinstance(server, dict) else {}
    except Exception:
        return {}


def resolve_timeout_model(server_config: dict | None = None) -> TimeoutModel:
    """Resolve the model from an explicit config, else config.yaml, else defaults."""
    if server_config is not None:
        return timeout_model_from_config(server_config)
    return timeout_model_from_config(load_server_config())


def resolve_snapshot_dir(server_config: dict | None = None) -> Path:
    """Directory holding saved snapshots (``session_slot_save_path`` root)."""
    config = server_config if server_config is not None else load_server_config()
    configured = config.get("session_slot_save_path")
    if configured:
        return Path(str(configured))
    return PROJECT_ROOT / "slot-cache"


# ---------------------------------------------------------------------------
# Snapshot-size cross-check (kv_memory_table.py)
# ---------------------------------------------------------------------------


def expected_snapshot_bytes(prompt_tokens: int) -> int:
    """Expected on-disk snapshot size for a *prompt_tokens* context.

    Formula, cross-checked against ``proxy/benchmarks/kv_memory_table.py``::

        bytes = KV_MIB_PER_TOKEN * prompt_tokens * 1 MiB

    ``KV_MIB_PER_TOKEN`` is the measured q8_0 per-total-ctx-cell KV footprint
    (1362.7 MiB / 131 072 cells) used by the F4 headroom table. The snapshot
    also carries the hybrid/recurrent state, so this is an order-of-magnitude
    cross-check, not an exact prediction — hence
    :data:`SNAPSHOT_BYTES_TOLERANCE`.
    """
    return int(KV_MIB_PER_TOKEN * max(0, int(prompt_tokens)) * MIB)


#: Accepted measured/expected ratio for the size cross-check. Deliberately
#: generous (a factor of two either way): quantisation, the hybrid/recurrent
#: state and per-layer overhead move the real file away from the pure-KV
#: estimate, but a 10x deviation means the snapshot is not what we think it is.
SNAPSHOT_BYTES_TOLERANCE: tuple[float, float] = (0.5, 2.0)


def snapshot_size_in_tolerance(
    measured_bytes: int | None,
    prompt_tokens: int,
    tolerance: tuple[float, float] = SNAPSHOT_BYTES_TOLERANCE,
) -> bool | None:
    """Whether *measured_bytes* is within *tolerance* of the expected size."""
    if measured_bytes is None:
        return None
    expected = expected_snapshot_bytes(prompt_tokens)
    if expected <= 0:
        return None
    ratio = measured_bytes / expected
    low, high = tolerance
    return low <= ratio <= high


# ---------------------------------------------------------------------------
# HTTP client abstraction (stubbed in the offline tests)
# ---------------------------------------------------------------------------


@dataclass
class SlotResponse:
    """Minimal HTTP response: status code plus decoded body."""

    status_code: int
    body: Any = None

    def text(self) -> str:
        if isinstance(self.body, str):
            return self.body
        try:
            return json.dumps(self.body)
        except (TypeError, ValueError):
            return str(self.body)

    def json_body(self) -> dict | None:
        return self.body if isinstance(self.body, dict) else None


class SlotStream(Protocol):
    """A streaming completion held open to keep another slot busy."""

    async def start(self) -> None: ...

    async def read(self) -> bytes | str | None: ...

    async def close(self) -> None: ...


class BenchmarkHTTPClient(Protocol):
    """The small HTTP surface the harness drives."""

    async def post(
        self, url: str, payload: dict, timeout: float
    ) -> SlotResponse: ...

    def open_stream(
        self, url: str, payload: dict, timeout: float
    ) -> SlotStream: ...


class _HttpxStream:
    def __init__(self, client, url: str, payload: dict, timeout: float) -> None:
        self._client = client
        self._url = url
        self._payload = payload
        self._timeout = timeout
        self._context = None
        self._response = None
        self._lines = None
        self._closed = False

    async def start(self) -> None:
        self._context = self._client.stream(
            "POST", self._url, json=self._payload, timeout=self._timeout
        )
        self._response = await self._context.__aenter__()

    async def read(self):
        if self._response is None:
            return None
        if self._lines is None:
            self._lines = self._response.aiter_lines()
        try:
            return await self._lines.__anext__()
        except StopAsyncIteration:
            return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.__aexit__(None, None, None)

    @property
    def closed(self) -> bool:
        return self._closed


class HttpxBenchmarkClient:
    """Live client: one ``httpx.AsyncClient`` for saves/restores and the load."""

    def __init__(self, timeout: float | None = None) -> None:
        if httpx is None:  # pragma: no cover - live-only failure
            raise RuntimeError(
                "httpx is required for a live run: pip install httpx"
            )
        # The client default must not be *lower* than the largest per-request
        # timeout, or the client silently clips a deliberately generous budget
        # (a populate issued with POPULATE_TIMEOUT_SECONDS would still abort at
        # the client default). One minute of margin keeps the per-request value
        # the binding one.
        effective = (
            float(timeout)
            if timeout is not None
            else max(600.0, POPULATE_TIMEOUT_SECONDS + 60.0)
        )
        self._client = httpx.AsyncClient(timeout=effective)

    async def post(self, url: str, payload: dict, timeout: float) -> SlotResponse:
        response = await self._client.post(url, json=payload, timeout=timeout)
        try:
            body = response.json()
        except Exception:
            body = response.text
        return SlotResponse(status_code=response.status_code, body=body)

    def open_stream(self, url: str, payload: dict, timeout: float) -> SlotStream:
        return _HttpxStream(self._client, url, payload, timeout)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

#: Outcomes a save/restore action can record. ``skipped`` means the harness
#: deliberately did not issue the call (restore after a failed save);
#: ``dry_run`` is a plan entry, never a measurement. ``not_requested`` marks an
#: optional step (the slot populate) the run did not ask for.
OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_NOT_REQUESTED = "not_requested"

#: Timeout for the *populate* completion (``--populate-slot``). It is a normal
#: chat completion, not a save/restore, so it is deliberately generous: the
#: adaptive slot timeout (60 s cap) would fail a 60k-token prefill that the
#: production dispatch path happily waits minutes for. Sizing: the production
#: host prefills this prompt shape at ~114 tokens/s (measured
#: 2026-10-28T10:39Z: 9 942 tokens in 87.6 s), so a 60k band needs ~530 s.
#: 900 leaves headroom for a host that is also serving live traffic.
POPULATE_TIMEOUT_SECONDS = 900.0

#: Timeout for the slot *erase* that precedes a populate. Administrative, not
#: part of the measured pair, and measured itself at ~14 ms for 94 250 tokens of
#: cached KV on an idle host — so anything above a second is queueing behind an
#: in-flight task elsewhere. Bounding it by the band's adaptive save/restore
#: timeout turned a 20 ms administrative step into a recorded ``failed`` erase
#: (observed at 18.02 s against the 10k band's 18.0 s budget) and made band
#: isolation look weaker than it was.
#:
#: It must nevertheless be generous, because abandoning an erase is *not*
#: harmless: the request stays queued server-side and can be executed later,
#: after the populate, wiping the very KV the band just wrote. A 120 s bound did
#: exactly that on a loaded host (erase recorded ``failed`` at 120.10 s, the
#: 60k+ populate then succeeded, and the save that followed captured
#: ``n_saved: 0``). 600 s only abandons an erase on a genuinely stuck server,
#: and the contention case is still visible as a large ``erase_seconds``.
ERASE_TIMEOUT_SECONDS = 600.0

#: How many times the *administrative* erase+populate phase may be attempted.
#: The save/restore pair is deliberately single-shot (a retry would hide the
#: timeout being measured), but the populate is not the measurement: it only
#: puts the band's KV into the slot. On the production host the proxy's model
#: health loop can recycle the llama-server instance during a long prefill
#: (``/slots`` stops answering while the GPU prefills, the probe times out, and
#: recovery unloads/reloads the model). The first full run lost two bands to
#: exactly that — a populate that ended in ``HTTP 500: proxy error: Failed to
#: read connection`` after 600 s. Retrying the populate cannot flatter the
#: save/restore numbers; it only stops an unrelated reload from discarding the
#: band.
POPULATE_ATTEMPTS = 3

#: Relative tolerance for "the snapshot holds this band's prompt". A saved KV
#: whose ``n_saved`` is outside this band of the populated prompt size is not
#: measuring the band: it describes whatever replaced the slot (a reload, or
#: another session taking the slot). The first full run produced exactly this
#: silently — one under-load band saved 3 878 KV cells for a 40k prompt and
#: still reported a "success".
SNAPSHOT_TOKEN_TOLERANCE = 0.10

#: Completion tokens produced by the populate and verification completions.
#: Kept tiny so the saved KV is dominated by the prompt itself.
PROBE_MAX_TOKENS = 8


#: The record schema (AC3): every field below is present in the JSON output.
RECORD_FIELDS: tuple[str, ...] = (
    "band",
    "target_tokens",
    "session_id",
    "slot_id",
    "salt",
    "prompt_tokens_estimated",
    "prompt_total_tokens",
    "prompt_processed_tokens",
    "cached_tokens",
    "cached_ratio",
    "populate_seconds",
    "populate_outcome",
    "populate_error",
    "populate_prompt_total_tokens",
    "populate_attempts",
    "erase_seconds",
    "erase_outcome",
    "timeout_seconds",
    "save_seconds",
    "save_outcome",
    "save_error",
    "save_within_timeout",
    "save_server_ms",
    "save_tokens",
    "restore_seconds",
    "restore_outcome",
    "restore_error",
    "restore_within_timeout",
    "restore_server_ms",
    "restore_tokens",
    "snapshot_bytes",
    "snapshot_bytes_on_disk",
    "expected_snapshot_bytes",
    "snapshot_size_ratio",
    "snapshot_size_in_tolerance",
    "snapshot_holds_band",
    "under_load",
    "verified",
)


@dataclass
class SnapshotRecord:
    """One band's save/restore measurement."""

    band: str
    target_tokens: int
    session_id: str
    slot_id: int
    salt: str
    prompt_tokens_estimated: int
    prompt_total_tokens: int
    prompt_processed_tokens: int | None
    cached_tokens: int | None
    cached_ratio: float | None
    populate_seconds: float | None
    populate_outcome: str
    populate_error: str | None
    populate_prompt_total_tokens: int | None
    timeout_seconds: float
    save_seconds: float | None
    save_outcome: str
    save_error: str | None
    save_within_timeout: bool | None
    save_server_ms: float | None
    save_tokens: int | None
    restore_seconds: float | None
    restore_outcome: str
    restore_error: str | None
    restore_within_timeout: bool | None
    restore_server_ms: float | None
    restore_tokens: int | None
    snapshot_bytes: int | None
    snapshot_bytes_on_disk: int | None
    expected_snapshot_bytes: int
    snapshot_size_ratio: float | None
    snapshot_size_in_tolerance: bool | None
    snapshot_holds_band: bool | None = None
    under_load: bool = False
    verified: bool = False
    populate_attempts: int = 0
    erase_seconds: float | None = None
    erase_outcome: str = OUTCOME_NOT_REQUESTED

    def to_json(self) -> dict[str, Any]:
        """Serialise to the documented record schema."""
        return {
            "band": self.band,
            "target_tokens": self.target_tokens,
            "session_id": self.session_id,
            "slot_id": self.slot_id,
            "salt": self.salt,
            "prompt_tokens_estimated": self.prompt_tokens_estimated,
            "prompt_total_tokens": self.prompt_total_tokens,
            "prompt_processed_tokens": self.prompt_processed_tokens,
            "cached_tokens": self.cached_tokens,
            "cached_ratio": (
                None if self.cached_ratio is None else round(self.cached_ratio, 4)
            ),
            "populate_seconds": (
                None
                if self.populate_seconds is None
                else round(self.populate_seconds, 3)
            ),
            "populate_outcome": self.populate_outcome,
            "populate_error": self.populate_error,
            "populate_prompt_total_tokens": self.populate_prompt_total_tokens,
            "populate_attempts": self.populate_attempts,
            "erase_seconds": (
                None if self.erase_seconds is None else round(self.erase_seconds, 3)
            ),
            "erase_outcome": self.erase_outcome,
            "timeout_seconds": round(self.timeout_seconds, 3),
            "save_seconds": (
                None if self.save_seconds is None else round(self.save_seconds, 3)
            ),
            "save_outcome": self.save_outcome,
            "save_error": self.save_error,
            "save_within_timeout": self.save_within_timeout,
            "save_server_ms": (
                None if self.save_server_ms is None else round(self.save_server_ms, 3)
            ),
            "save_tokens": self.save_tokens,
            "restore_seconds": (
                None if self.restore_seconds is None else round(self.restore_seconds, 3)
            ),
            "restore_outcome": self.restore_outcome,
            "restore_error": self.restore_error,
            "restore_within_timeout": self.restore_within_timeout,
            "restore_server_ms": (
                None
                if self.restore_server_ms is None
                else round(self.restore_server_ms, 3)
            ),
            "restore_tokens": self.restore_tokens,
            "snapshot_bytes": self.snapshot_bytes,
            "snapshot_bytes_on_disk": self.snapshot_bytes_on_disk,
            "expected_snapshot_bytes": self.expected_snapshot_bytes,
            "snapshot_size_ratio": (
                None
                if self.snapshot_size_ratio is None
                else round(self.snapshot_size_ratio, 4)
            ),
            "snapshot_size_in_tolerance": self.snapshot_size_in_tolerance,
            "snapshot_holds_band": self.snapshot_holds_band,
            "under_load": self.under_load,
            "verified": self.verified,
        }


@dataclass
class BandPlan:
    """What a band will do, computed without any HTTP (dry-run output)."""

    band: str
    target_tokens: int
    prompt_tokens_estimated: int
    timeout_seconds: float
    expected_snapshot_bytes: int
    under_load: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "target_tokens": self.target_tokens,
            "prompt_tokens_estimated": self.prompt_tokens_estimated,
            "timeout_seconds": round(self.timeout_seconds, 3),
            "expected_snapshot_bytes": self.expected_snapshot_bytes,
            "under_load": self.under_load,
        }


@dataclass
class BenchmarkResult:
    """A whole run: the plan, the records, and derived totals."""

    dry_run: bool
    under_load: bool
    verified: bool
    base_url: str
    timeout_model: TimeoutModel
    plan: list[BandPlan] = field(default_factory=list)
    records: list[SnapshotRecord] = field(default_factory=list)
    generated_at: str = ""
    populate_slot: bool = False
    slot_id: int = 0
    verify_slot_id: int | None = None
    load_slot_id: int | None = None
    populate_timeout_seconds: float = POPULATE_TIMEOUT_SECONDS

    def totals(self) -> dict[str, Any]:
        save_success = sum(
            1 for r in self.records if r.save_outcome == OUTCOME_SUCCESS
        )
        restore_success = sum(
            1 for r in self.records if r.restore_outcome == OUTCOME_SUCCESS
        )
        measured = [r for r in self.records if r.snapshot_bytes is not None]
        verified = [r for r in self.records if r.cached_tokens is not None]
        return {
            "bands": len(self.records),
            "save_success": save_success,
            "save_failed": sum(
                1 for r in self.records if r.save_outcome == OUTCOME_FAILED
            ),
            "restore_success": restore_success,
            "restore_failed": sum(
                1 for r in self.records if r.restore_outcome == OUTCOME_FAILED
            ),
            "restore_skipped": sum(
                1 for r in self.records if r.restore_outcome == OUTCOME_SKIPPED
            ),
            "saves_within_timeout": sum(
                1 for r in self.records if r.save_within_timeout
            ),
            "restores_within_timeout": sum(
                1 for r in self.records if r.restore_within_timeout
            ),
            "snapshots_measured": len(measured),
            "snapshots_out_of_tolerance": sum(
                1
                for r in self.records
                if r.snapshot_size_in_tolerance is False
            ),
            "populated_bands": sum(
                1 for r in self.records if r.populate_outcome == OUTCOME_SUCCESS
            ),
            "snapshots_not_holding_band": sum(
                1 for r in self.records if r.snapshot_holds_band is False
            ),
            "verified_bands": len(verified),
            "mean_cached_ratio": (
                round(
                    sum(r.cached_ratio for r in verified) / len(verified), 4
                )
                if verified
                else None
            ),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "slot_snapshot_benchmark",
            "schema_version": 3,
            "sentinel": SNAPSHOT_BENCHMARK_SENTINEL,
            "generated_at": self.generated_at,
            "dry_run": self.dry_run,
            "under_load": self.under_load,
            "verify_chat": self.verified,
            "populate_slot": self.populate_slot,
            "slot_id": self.slot_id,
            "verify_slot_id": self.verify_slot_id,
            "load_slot_id": self.load_slot_id,
            "populate_timeout_seconds": self.populate_timeout_seconds,
            "base_url": self.base_url,
            "hit_definition": "cached_tokens = prompt_total_tokens - timings.prompt_n",
            "timeout_model": {
                "base_seconds": self.timeout_model.base_seconds,
                "per_token_seconds": self.timeout_model.per_token_seconds,
                "cap_seconds": self.timeout_model.cap_seconds,
                "cap_tokens": round(self.timeout_model.cap_tokens, 1),
            },
            "bands": [p.to_json() for p in self.plan],
            "records": [r.to_json() for r in self.records],
            "totals": self.totals(),
        }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadSpec:
    """A second streaming session that keeps another slot processing.

    ``id_slot`` pins the load stream to a specific llama-server slot (the one
    *not* being measured). Without it llama-server assigns by availability and
    the stream can land on the measured slot, which stops being "another
    slot" and turns the save into a same-slot contention test.
    """

    messages: tuple[dict[str, Any], ...] = (
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Write a very long, detailed explanation of how a computer "
                "counts from one to one thousand. Do not stop early."
            ),
        },
    )
    model: str = "Qwen3"
    max_tokens: int = 512
    id_slot: int | None = None

    def payload(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [dict(m) for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if self.id_slot is not None:
            payload["id_slot"] = self.id_slot
        return payload


def _truncate(text: str, limit: int = 300) -> str:
    flattened = " ".join(str(text).split())
    return flattened if len(flattened) <= limit else flattened[:limit] + "…"


def _retryable_populate_error(error: str | None) -> bool:
    """Whether a failed populate is worth re-issuing.

    A transport failure, a read timeout or a 5xx means the server never gave a
    verdict on the request (typically because the proxy's model-health loop
    recycled the instance during the long prefill). A 4xx is deterministic —
    the same request would fail again — so it is reported, not retried.
    """
    if not error:
        return False
    if error.startswith("HTTP "):
        return error.startswith("HTTP 5")
    return True


def _snapshot_holds_band(
    save_tokens: int | None, band_tokens: int | None
) -> bool | None:
    """Whether the saved KV actually holds *band_tokens* of prompt.

    ``n_saved`` is the number of KV cells the save wrote. Comparing it with the
    populated prompt size is the only cheap way to tell "this snapshot is the "
    "band" from "this snapshot is whatever replaced the slot": a reload, or a
    concurrent session taking the slot, silently yields a *successful* save of
    the wrong KV. Returns ``None`` when either number is unavailable.

    ``n_saved == 0`` is a **failure**, not missing data: it means the slot was
    empty when the save ran (observed when a timed-out erase executed late and
    wiped the band).
    """
    if save_tokens is None or not band_tokens:
        return None
    return abs(int(save_tokens) - int(band_tokens)) <= max(
        1, int(SNAPSHOT_TOKEN_TOLERANCE * int(band_tokens))
    )


def _numeric(body: dict[str, Any] | None, *path: str) -> float | None:
    """Read a numeric value from a (possibly absent) nested response field."""
    node: Any = body
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return float(node) if isinstance(node, (int, float)) else None


def _integer(body: dict[str, Any] | None, key: str) -> int | None:
    """Read an integer value from a (possibly absent) response field."""
    if not isinstance(body, dict):
        return None
    value = body.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _slot_url(base_url: str, slot_id: int, action: str) -> str:
    return f"{base_url.rstrip('/')}/slots/{slot_id}?action={action}"


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _format_error(exc: BaseException) -> str:
    detail = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


async def _issue_slot_action(
    client: BenchmarkHTTPClient,
    base_url: str,
    slot_id: int,
    action: str,
    filename: str,
    timeout: float,
    model: str | None = None,
) -> tuple[float, str, str | None, dict[str, Any] | None]:
    """Issue *action* exactly once; return ``(seconds, outcome, error, body)``.

    Deliberately single-shot: a retry would mask the timeout this benchmark
    exists to measure (AC3: a failure is recorded, never retried away).
    ``body`` is the decoded JSON response on success — llama-server reports
    ``n_written`` (snapshot bytes) and ``timings.save_ms/restore_ms`` there,
    which are more authoritative than a filesystem stat racing the write.

    Passing *model* puts the model name in the request body, which the
    llama-server router requires to forward the action to the loaded instance.
    The router is the production path: the proxy issues its own saves through
    ``http://localhost:{llama_server_port}/slots/{id}?action=…`` carrying
    ``model`` (``proxy/proxy/session.py:_call_slot_endpoint``). It also makes
    the measurement immune to instance port churn, since the router keeps
    serving on its fixed port while the per-model instances are recycled onto
    new ephemeral ports. A direct instance ignores the field.
    """
    payload: dict[str, Any] = {"filename": Path(filename).name}
    if model:
        payload["model"] = model
    started = time.monotonic()
    try:
        response = await client.post(
            _slot_url(base_url, slot_id, action), payload, timeout
        )
    except Exception as exc:
        return time.monotonic() - started, OUTCOME_FAILED, _format_error(exc), None
    elapsed = time.monotonic() - started
    if getattr(response, "status_code", None) != 200:
        return (
            elapsed,
            OUTCOME_FAILED,
            f"HTTP {response.status_code}: {_truncate(response.text())}",
            None,
        )
    return elapsed, OUTCOME_SUCCESS, None, response.json_body()


async def _erase_slot(
    client: BenchmarkHTTPClient,
    base_url: str,
    slot_id: int,
    timeout: float,
    model: str | None = None,
) -> tuple[float, str, str | None]:
    """Clear *slot_id*'s KV before a band populates it (best-effort).

    The erase makes each band's save/restore describe only that band's KV:
    without it a leftover prefix from the previous band can survive into the
    next band's populate and its restore, silently coupling the measurements.

    *model* is forwarded when set, so the erase reaches the instance behind
    the router the same way the save/restore pair does.
    """
    payload: dict[str, Any] = {}
    if model:
        payload["model"] = model
    started = time.monotonic()
    try:
        response = await client.post(
            _slot_url(base_url, slot_id, "erase"), payload, timeout
        )
    except Exception as exc:
        return time.monotonic() - started, OUTCOME_FAILED, _format_error(exc)
    elapsed = time.monotonic() - started
    if getattr(response, "status_code", None) != 200:
        return (
            elapsed,
            OUTCOME_FAILED,
            f"HTTP {response.status_code}: {_truncate(response.text())}",
        )
    return elapsed, OUTCOME_SUCCESS, None


async def _completion_tokens(
    client: BenchmarkHTTPClient,
    base_url: str,
    prompt: str,
    model: str,
    timeout: float,
    id_slot: int | None = None,
) -> tuple[
    int | None, int | None, float, str, str | None
]:
    """One non-streaming completion; return the token and outcome detail.

    Returns ``(prompt_total, prompt_n, seconds, outcome, error)`` where
    ``prompt_total`` is ``usage.prompt_tokens`` and ``prompt_n`` is
    llama-server's ``timings.prompt_n`` (prompt tokens actually processed).
    Both are ``None`` when the server does not report them.

    ``id_slot`` pins the completion to a specific slot. That matters for the
    measurement: a populate completion must fill the slot the save will
    capture, and a verification completion can only report restored-prefix
    reuse if it lands on the slot that was just restored (llama-server assigns
    by availability otherwise).
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": 0.0,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot
    started = time.monotonic()
    try:
        response = await client.post(_chat_url(base_url), payload, timeout)
    except Exception as exc:
        return None, None, time.monotonic() - started, OUTCOME_FAILED, _format_error(exc)
    elapsed = time.monotonic() - started
    if getattr(response, "status_code", None) != 200:
        return (
            None,
            None,
            elapsed,
            OUTCOME_FAILED,
            f"HTTP {response.status_code}: {_truncate(response.text())}",
        )
    body = response.json_body() or {}
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    total = usage.get("prompt_tokens")
    processed = timings.get("prompt_n")
    return (
        int(total) if isinstance(total, (int, float)) else None,
        int(processed) if isinstance(processed, (int, float)) else None,
        elapsed,
        OUTCOME_SUCCESS,
        None,
    )


async def _verification_tokens(
    client: BenchmarkHTTPClient,
    base_url: str,
    prompt: str,
    model: str,
    timeout: float,
    id_slot: int | None = None,
) -> tuple[int | None, int | None]:
    """Verification completion: ``(prompt_total, prompt_n)``, ignoring outcome detail."""
    total, processed, _seconds, outcome, _error = await _completion_tokens(
        client, base_url, prompt, model, timeout, id_slot
    )
    if outcome != OUTCOME_SUCCESS:
        return None, None
    return total, processed


async def _load_driver(
    client: BenchmarkHTTPClient,
    base_url: str,
    load: LoadSpec,
    stop: asyncio.Event,
    timeout: float,
    opened: asyncio.Event | None = None,
) -> None:
    """Keep a streaming completion open on another slot until *stop* is set."""
    url = _chat_url(base_url)
    while not stop.is_set():
        stream = client.open_stream(url, load.payload(), timeout)
        try:
            await stream.start()
            if opened is not None:
                opened.set()
            while not stop.is_set():
                chunk = await stream.read()
                if chunk is None:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed load stream weakens the "under load" claim but must not
            # destroy the measurement in flight; give the slot a moment and
            # re-open instead of aborting the band.
            await asyncio.sleep(0.5)
        finally:
            with contextlib.suppress(Exception):
                await stream.close()


def _safe_filename_token(value: str) -> str:
    """Restrict *value* to characters safe in a snapshot filename."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)


def _snapshot_path(snapshot_dir: Path | None, filename: str) -> Path | None:
    if snapshot_dir is None:
        return None
    return Path(snapshot_dir) / Path(filename).name


def _within_timeout(
    outcome: str, seconds: float | None, timeout: float
) -> bool | None:
    """Whether a *successful* action finished inside *timeout*.

    A failed action is never "within timeout" (it did not complete), and a
    skipped action has no latency at all.
    """
    if seconds is None:
        return None
    return outcome == OUTCOME_SUCCESS and seconds <= timeout


async def measure_band(
    band: BandSpec,
    *,
    client: BenchmarkHTTPClient,
    base_url: str,
    slot_id: int,
    session_id: str,
    salt: str,
    model: str,
    timeout_model: TimeoutModel,
    snapshot_dir: Path | None = None,
    verify_chat: bool = False,
    load: LoadSpec | None = None,
    populate_slot: bool = False,
    verify_slot_id: int | None = None,
) -> SnapshotRecord:
    """Measure one band's save/restore pair.

    With *load* set, a streaming completion is kept open on another slot for
    the whole save/restore window, so the saves issue while another slot is
    processing (the parent's AC1 "under load" clause).

    With *populate_slot*, the band prompt is first prefilled **into the measured
    slot** (one pinned completion). That is what makes the numbers production-
    shaped: a save then captures a prompt-sized KV rather than an empty slot
    (and a restore then has a prompt-sized snapshot to bring back). Without it
    ``snapshot_bytes`` describes an almost-empty slot and ``cached_tokens``
    cannot say anything about restore effectiveness.

    *verify_slot_id* pins the verification completion (defaults to unpinned, the
    historical best-effort behaviour; pinning it to the measured slot is what
    makes ``timings.prompt_n`` authoritative).
    """
    prompt = band_prompt(band, salt)
    estimated = estimated_prompt_tokens(prompt)
    timeout = timeout_model.timeout_for(estimated)
    filename = (
        f"snapshot-bench-{_safe_filename_token(band.name)}"
        f"-{_safe_filename_token(salt)}.bin"
    )

    populate_total: int | None = None
    populate_seconds: float | None = None
    populate_outcome = OUTCOME_NOT_REQUESTED
    populate_error: str | None = None
    populate_attempts = 0
    erase_seconds: float | None = None
    erase_outcome = OUTCOME_NOT_REQUESTED
    if populate_slot:
        for attempt in range(1, POPULATE_ATTEMPTS + 1):
            populate_attempts = attempt
            # Clear the slot first: a leftover prefix from the previous band
            # could otherwise survive into this band's populate and restore,
            # coupling the bands together. Best-effort — a failed erase does
            # not stop the measurement, it only means the isolation claim is
            # weaker.
            erase_seconds, erase_outcome, _erase_error = await _erase_slot(
                client, base_url, slot_id, ERASE_TIMEOUT_SECONDS, model
            )
            (
                populate_total,
                _populate_processed,
                populate_seconds,
                populate_outcome,
                populate_error,
            ) = await _completion_tokens(
                client,
                base_url,
                prompt,
                model,
                POPULATE_TIMEOUT_SECONDS,
                slot_id,
            )
            if populate_outcome == OUTCOME_SUCCESS:
                break
            if not _retryable_populate_error(populate_error):
                break
            # A transport failure or a 5xx means the request never got a
            # verdict, or the instance was recycled mid-prefill — retrying the
            # *administrative* populate cannot flatter the save/restore numbers
            # it prepares. A 4xx (e.g. context-too-large) is deterministic and
            # is not retried.

    stop = asyncio.Event()
    opened = asyncio.Event()
    driver: asyncio.Task | None = None
    if load is not None:
        driver = asyncio.create_task(
            _load_driver(client, base_url, load, stop, 600.0, opened)
        )
        # Wait until the streaming request has actually reached the server
        # before the first save, so "under load" is a real condition, not a
        # scheduling race. Best-effort: a stream that never opens (e.g. no
        # free slot) must not hang the run — the record simply measures
        # without the intended load and keeps under_load=false semantics for
        # the save interleaving.
        try:
            await asyncio.wait_for(opened.wait(), timeout=15.0)
        except TimeoutError:
            pass

    try:
        (
            save_seconds,
            save_outcome,
            save_error,
            save_body,
        ) = await _issue_slot_action(
            client, base_url, slot_id, "save", filename, timeout, model
        )

        # llama-server reports the authoritative size and server-side time in
        # the response body. The filesystem stat is kept as corroboration:
        # statting late (the old behaviour) could read a *previous* band's file
        # and mis-report the footprint.
        save_server_ms = _numeric(save_body, "timings", "save_ms")
        save_tokens = _integer(save_body, "n_saved")
        snapshot_file = _snapshot_path(snapshot_dir, filename)
        snapshot_bytes_on_disk = None
        if save_outcome == OUTCOME_SUCCESS and snapshot_file is not None:
            try:
                snapshot_bytes_on_disk = snapshot_file.stat().st_size
            except OSError:
                snapshot_bytes_on_disk = None
        snapshot_bytes = _integer(save_body, "n_written") or snapshot_bytes_on_disk

        restore_server_ms: float | None = None
        restore_tokens: int | None = None
        if save_outcome == OUTCOME_SUCCESS:
            (
                restore_seconds,
                restore_outcome,
                restore_error,
                restore_body,
            ) = await _issue_slot_action(
                client, base_url, slot_id, "restore", filename, timeout, model
            )
            restore_server_ms = _numeric(restore_body, "timings", "restore_ms")
            restore_tokens = _integer(restore_body, "n_restored")
        else:
            # No snapshot was written: issuing a restore would measure nothing
            # and would add a spurious failure to the record.
            restore_seconds = None
            restore_outcome = OUTCOME_SKIPPED
            restore_error = "save failed; restore not issued"

        prompt_total: int | None = None
        prompt_n: int | None = None
        if verify_chat and restore_outcome == OUTCOME_SUCCESS:
            prompt_total, prompt_n = await _verification_tokens(
                client, base_url, prompt, model, timeout, verify_slot_id
            )
    finally:
        stop.set()
        if driver is not None:
            driver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver

    if prompt_total is None:
        prompt_total = populate_total
    if prompt_total is None:
        prompt_total = estimated
    cached_tokens = None
    cached_ratio = None
    if prompt_n is not None:
        cached_tokens = max(0, prompt_total - prompt_n)
        cached_ratio = (
            cached_tokens / prompt_total if prompt_total > 0 else None
        )

    expected_bytes = expected_snapshot_bytes(prompt_total)

    return SnapshotRecord(
        band=band.name,
        target_tokens=band.target_tokens,
        session_id=session_id,
        slot_id=slot_id,
        salt=salt,
        prompt_tokens_estimated=estimated,
        prompt_total_tokens=prompt_total,
        prompt_processed_tokens=prompt_n,
        cached_tokens=cached_tokens,
        cached_ratio=cached_ratio,
        populate_seconds=populate_seconds,
        populate_outcome=populate_outcome,
        populate_error=populate_error,
        populate_prompt_total_tokens=populate_total,
        populate_attempts=populate_attempts,
        timeout_seconds=timeout,
        save_seconds=save_seconds,
        save_outcome=save_outcome,
        save_error=save_error,
        save_within_timeout=_within_timeout(
            save_outcome, save_seconds, timeout
        ),
        save_server_ms=save_server_ms,
        save_tokens=save_tokens,
        restore_seconds=restore_seconds,
        restore_outcome=restore_outcome,
        restore_error=restore_error,
        restore_within_timeout=_within_timeout(
            restore_outcome, restore_seconds, timeout
        ),
        restore_server_ms=restore_server_ms,
        restore_tokens=restore_tokens,
        snapshot_bytes=snapshot_bytes,
        snapshot_bytes_on_disk=snapshot_bytes_on_disk,
        snapshot_holds_band=_snapshot_holds_band(
            save_tokens, populate_total or prompt_total
        ),
        expected_snapshot_bytes=expected_bytes,
        snapshot_size_ratio=(
            None if snapshot_bytes is None else snapshot_bytes / expected_bytes
            if expected_bytes > 0
            else None
        ),
        snapshot_size_in_tolerance=snapshot_size_in_tolerance(
            snapshot_bytes, prompt_total
        ),
        under_load=load is not None,
        verified=prompt_n is not None,
        erase_seconds=erase_seconds,
        erase_outcome=erase_outcome,
    )


def plan_for(
    bands: Sequence[BandSpec],
    *,
    salt: str,
    timeout_model: TimeoutModel,
    under_load: bool,
) -> list[BandPlan]:
    """Compute the dry-run plan: sizing and timeout per band, no HTTP."""
    plans = []
    for band in bands:
        estimated = estimated_prompt_tokens(band_prompt(band, salt))
        plans.append(
            BandPlan(
                band=band.name,
                target_tokens=band.target_tokens,
                prompt_tokens_estimated=estimated,
                timeout_seconds=timeout_model.timeout_for(estimated),
                expected_snapshot_bytes=expected_snapshot_bytes(estimated),
                under_load=under_load,
            )
        )
    return plans


def default_salt() -> str:
    """A salt unique to this run (bands stay cold across runs)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def run_benchmark(
    *,
    bands: Sequence[BandSpec] = DEFAULT_BANDS,
    base_url: str = "http://localhost:8080",
    slot_id: int = 0,
    session_id: str = "snapshot-benchmark",
    salt: str | None = None,
    model: str = "Qwen3",
    timeout_model: TimeoutModel = DEFAULT_TIMEOUT_MODEL,
    snapshot_dir: Path | None = None,
    dry_run: bool = False,
    under_load: bool = False,
    verify_chat: bool = False,
    load: LoadSpec | None = None,
    client: BenchmarkHTTPClient | None = None,
    populate_slot: bool = False,
    verify_slot_id: int | None = None,
) -> BenchmarkResult:
    """Run (or plan, when *dry_run*) every band and return the result."""
    run_salt = salt or default_salt()
    result = BenchmarkResult(
        dry_run=dry_run,
        under_load=under_load,
        verified=verify_chat,
        base_url=base_url,
        timeout_model=timeout_model,
        plan=plan_for(
            bands, salt=run_salt, timeout_model=timeout_model, under_load=under_load
        ),
        generated_at=datetime.now(UTC).isoformat(),
        populate_slot=populate_slot,
        slot_id=slot_id,
        verify_slot_id=verify_slot_id,
        load_slot_id=load.id_slot if load is not None else None,
    )
    if dry_run:
        return result

    own_client = client is None
    active = client if client is not None else HttpxBenchmarkClient()
    load_spec = load
    if under_load and load_spec is None:
        load_spec = LoadSpec(model=model)
    try:
        for band in bands:
            result.records.append(
                await measure_band(
                    band,
                    client=active,
                    base_url=base_url,
                    slot_id=slot_id,
                    session_id=f"{session_id}-{band.name}",
                    salt=run_salt,
                    model=model,
                    timeout_model=timeout_model,
                    snapshot_dir=snapshot_dir,
                    verify_chat=verify_chat,
                    load=load_spec,
                    populate_slot=populate_slot,
                    verify_slot_id=verify_slot_id,
                )
            )
    finally:
        if own_client and hasattr(active, "aclose"):
            await active.aclose()
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_text(result: BenchmarkResult) -> str:
    """Human-readable summary of a run (or plan)."""
    lines = []
    header = "Slot snapshot save/restore benchmark"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(
        "timeout model: min("
        f"{result.timeout_model.base_seconds} + "
        f"{result.timeout_model.per_token_seconds} * tokens, "
        f"{result.timeout_model.cap_seconds}) "
        f"— capped from {result.timeout_model.cap_tokens:.0f} tokens"
    )
    lines.append(f"under load: {'yes' if result.under_load else 'no'}")
    lines.append(
        "populate slot before save: "
        f"{'yes' if result.populate_slot else 'no'}"
        + (
            f" (slot {result.slot_id}, verify pinned to "
            f"{result.verify_slot_id if result.verify_slot_id is not None else 'any'})"
            if result.populate_slot
            else ""
        )
    )
    lines.append("")
    if result.dry_run:
        lines.append("DRY RUN — no HTTP issued")
    lines.append(
        f"{'band':<6} {'target':>8} {'prompt≈':>9} {'timeout':>8} "
        f"{'fill s':>8} {'save s':>8} {'restore s':>10} {'bytes':>12} "
        f"{'cached':>8} outcome"
    )
    by_band = {r.band: r for r in result.records}
    for plan in result.plan:
        record = by_band.get(plan.band)
        if record is None:
            lines.append(
                f"{plan.band:<6} {plan.target_tokens:>8} "
                f"{plan.prompt_tokens_estimated:>9} "
                f"{plan.timeout_seconds:>8.1f} {'—':>8} {'—':>8} {'—':>10} "
                f"{plan.expected_snapshot_bytes:>12} {'—':>8} planned"
            )
            continue
        lines.append(
            f"{record.band:<6} {record.target_tokens:>8} "
            f"{record.prompt_tokens_estimated:>9} "
            f"{record.timeout_seconds:>8.1f} "
            f"{_fmt(record.populate_seconds):>8} "
            f"{_fmt(record.save_seconds):>8} "
            f"{_fmt(record.restore_seconds):>10} "
            f"{_fmt_bytes(record.snapshot_bytes):>12} "
            f"{_fmt_int(record.cached_tokens):>8} "
            f"{record.save_outcome}/{record.restore_outcome}"
        )
        for label, error in (
            ("populate", record.populate_error),
            ("save", record.save_error),
            ("restore", record.restore_error),
        ):
            if error:
                lines.append(f"       {label} error: {error}")
    lines.append("")
    lines.append(
        "hit definition: cached_tokens = prompt_total_tokens - timings.prompt_n "
        "(latency is corroboration only)"
    )
    totals = result.totals()
    if not result.dry_run:
        lines.append(
            "save ok/failed: {save_success}/{save_failed}; "
            "restore ok/failed/skipped: {restore_success}/{restore_failed}/"
            "{restore_skipped}; within timeout: save {saves_within_timeout}, "
            "restore {restores_within_timeout}".format(**totals)
        )
        lines.append(
            f"snapshots measured: {totals['snapshots_measured']} "
            f"(out of kv_memory_table tolerance: "
            f"{totals['snapshots_out_of_tolerance']})"
        )
        lines.append(f"populated bands: {totals['populated_bands']}")
        if totals["snapshots_not_holding_band"]:
            lines.append(
                "WARNING: {n} snapshot(s) do not hold the band's prompt "
                "(n_saved far from the populated prompt size): the slot was "
                "replaced between populate and save (a model reload, or "
                "another session taking the slot). Those bands describe "
                "something other than the band and must not be read as "
                "measurements.".format(n=totals["snapshots_not_holding_band"])
            )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_int(value: int | None) -> str:
    return "—" if value is None else str(value)


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


def write_result(result: BenchmarkResult, output_dir: Path | None = None) -> Path:
    """Persist the JSON result under *output_dir* and return the path."""
    directory = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"slot_snapshot_benchmark_{stamp}.json"
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure llama-server KV slot snapshot save/restore cost per "
            "prompt-size band against the adaptive timeout model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bands",
        default="10k,25k,40k,60k+",
        help=(
            "Comma-separated band names (10k,25k,40k,60k+) or token counts. "
            "Default: 10k,25k,40k,60k+"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (sizing, timeouts, expected bytes) without any HTTP",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="llama-server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--slot-id", type=int, default=0, help="Slot to save/restore (default: 0)"
    )
    parser.add_argument(
        "--session-id",
        default="snapshot-benchmark",
        help="Session-id prefix recorded per band (default: snapshot-benchmark)",
    )
    parser.add_argument(
        "--salt",
        default=None,
        help="Prompt salt (default: a UTC run stamp, so bands stay cold)",
    )
    parser.add_argument("--model", default="Qwen3", help="Model name for completions")
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help=(
            "Directory llama-server writes snapshots to (default: "
            "session_slot_save_path from config.yaml). Used only to stat the "
            "on-disk bytes; a missing file records null bytes rather than failing"
        ),
    )
    parser.add_argument(
        "--under-load",
        action="store_true",
        help=(
            "Keep a second streaming completion open on another slot for the "
            "whole save/restore window (AC1 'under load'); requires >= 2 slots"
        ),
    )
    parser.add_argument(
        "--load-max-tokens",
        type=int,
        default=512,
        help="Completion tokens for the under-load stream (default: 512)",
    )
    parser.add_argument(
        "--verify-chat",
        action="store_true",
        help=(
            "After each restore, send one completion and record "
            "timings.prompt_n so cached_tokens can be computed"
        ),
    )
    parser.add_argument(
        "--populate-slot",
        action="store_true",
        help=(
            "Prefill the band prompt into the measured slot before saving it, "
            "so the snapshot is prompt-sized and the restore has real KV to "
            "bring back (required for meaningful bytes/cached_tokens)"
        ),
    )
    parser.add_argument(
        "--verify-slot-id",
        type=int,
        default=None,
        help=(
            "Pin the verification completion to this slot (normally the "
            "measured --slot-id) so timings.prompt_n describes the restored slot"
        ),
    )
    parser.add_argument(
        "--load-slot-id",
        type=int,
        default=None,
        help=(
            "Pin the --under-load stream to this slot (normally a slot other "
            "than --slot-id) so 'another slot' is a real condition"
        ),
    )
    parser.add_argument(
        "--timeout-base",
        type=float,
        default=None,
        help="Override session_slot_timeout_seconds (default: from config)",
    )
    parser.add_argument(
        "--timeout-per-token",
        type=float,
        default=None,
        help="Override session_slot_timeout_per_token_seconds (default: from config)",
    )
    parser.add_argument(
        "--timeout-max",
        type=float,
        default=None,
        help="Override session_slot_max_timeout_seconds (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Result directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not persist the JSON result file (stdout only)",
    )
    return parser.parse_args(argv)


def _resolve_model_from_args(args: argparse.Namespace) -> TimeoutModel:
    model = resolve_timeout_model()
    if args.timeout_base is not None:
        model = TimeoutModel(args.timeout_base, model.per_token_seconds, model.cap_seconds)
    if args.timeout_per_token is not None:
        model = TimeoutModel(model.base_seconds, args.timeout_per_token, model.cap_seconds)
    if args.timeout_max is not None:
        model = TimeoutModel(
            model.base_seconds, model.per_token_seconds, args.timeout_max
        )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bands = parse_bands(args.bands)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    timeout_model = _resolve_model_from_args(args)
    snapshot_dir = (
        Path(args.snapshot_dir)
        if args.snapshot_dir
        else resolve_snapshot_dir()
    )
    result = asyncio.run(
        run_benchmark(
            bands=bands,
            base_url=args.base_url,
            slot_id=args.slot_id,
            session_id=args.session_id,
            salt=args.salt,
            model=args.model,
            timeout_model=timeout_model,
            snapshot_dir=snapshot_dir,
            dry_run=args.dry_run,
            under_load=args.under_load,
            verify_chat=args.verify_chat,
            populate_slot=args.populate_slot,
            verify_slot_id=args.verify_slot_id,
            load=(
                LoadSpec(
                    model=args.model,
                    max_tokens=args.load_max_tokens,
                    id_slot=args.load_slot_id,
                )
                if args.under_load
                else None
            ),
        )
    )

    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        print(render_text(result))

    if not args.dry_run and not args.no_write:
        path = write_result(result, Path(args.output_dir))
        if not args.json:
            print(f"\nresult written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
