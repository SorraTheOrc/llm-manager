"""Cold-regime replay proof harness (LP-0MW957TAE000QD9N).

Parent LP-0MW7V2D13004ZO65 AC3/AC10, test half. The pricing and the bound the
proof asserts are shipped: the cold-only calibration and the deadline token
capacity by LP-0MW957RB3000ER79, the abandoned-prefill cold-sample source by
LP-0MWGAEBWJ00334W2. This harness replays two **committed** fixtures through the
real calibration + admission path so the cold rate, the capacity refusals and
the 2026-11-03 mispricing are reproducible from the repository forever — no row
reads a live log or a live recording.

Fixtures
--------

``fixtures/cold_regime_window_2026_11_06.jsonl``
    The certified window (2026-11-06 14:15:50 -> 2026-11-08 00:00) reduced to
    the decisive rows. Three record kinds, each carrying an inline
    ``provenance`` string plus its raw ``evidence``:

    ``measured_cold``
        The window's genuinely cold *completed* prefills at the parent's cited
        3.81-4.26 ms/token. Every one is below the shipped
        ``COLD_MIN_PROCESSED_TOKENS`` floor, which is why the cold window needs
        the abandoned source at all; the rows are the AC1 rate oracle, not
        calibration input.
    ``abandoned``
        Prefills the deadline/ceiling abandoned. ``total``/``cache``/
        ``processed``/``time_ms`` are exactly the four ``prompt_progress`` fields
        the proxy observes, i.e. what
        :func:`proxy.prefill_eta.record_abandoned_prefill_sample` records. These
        rows calibrate the cold window in the replay.
    ``predicted_warm``
        Admissions that *predicted* a reused prefix (>= 0.98 reuse) yet were
        still prefilling at the deadline. Replayed on the honest basis
        (residency unmeasured => the whole prompt is priced cold), these are the
        AC2 refusal population.

``fixtures/recorded_sequence_2026_11_03.jsonl``
    The 2026-11-03 incident (LP-0MW0KZUUS002LICZ), **reconstructed** from the
    evidence recorded there: a 51,945-byte tool-role append priced at 173 new
    tokens while llama-server prefilled 13,006 (53.2 s to first byte), repeated
    at 72.0 s. The original recording directory
    ``~/.llm-proxy/session-recordings/herdr-1793695298-3654733-20033`` was
    pruned; the fixture header records that provenance and the pruned path.

What the replay exercises
-------------------------

* :func:`load_cold_window` feeds the abandoned rows through the **real**
  recorder, so the fitted rate is the shipped calibration path's result.
* :func:`replay_predicted_warm` feeds every predicted-warm row through the
  **real** admission decision (:func:`proxy.prefill_eta.evaluate_admission`)
  with the shipped ``proxy/config.yaml`` guard settings.
* :func:`replay_incident` prices the incident's appended turn through the
  **real** shipped large-append rule
  (:func:`proxy.provider.expected_cached_ratio`), which forces
  ``cached_ratio = 0`` above ``local_large_append_bytes_threshold``, and then
  admits/refuses it through the same decision.

:func:`reset` clears every piece of global state the replay touches
(calibration windows + the provider's session pricing state), so a test can
start from zero and each replay is deterministic and order-independent.

This module is test tooling: no production module may import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from proxy import prefill_eta, provider

MODEL = "Qwen3"
#: The pruned incident recording's session id (the fixture header names it).
SESSION = "herdr-1793695298-3654733-20033"

FIXTURES = Path(__file__).parent / "fixtures"
WINDOW_FIXTURE = FIXTURES / "cold_regime_window_2026_11_06.jsonl"
INCIDENT_FIXTURE = FIXTURES / "recorded_sequence_2026_11_03.jsonl"
#: ``proxy/config.yaml`` — the shipped guard settings the replay runs with.
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

#: The incident prompt sits 372 tokens BELOW the shipped 20 000-token pricing
#: floor, so the incident replay lowers the floor to this value to exercise the
#: cold-regime branch the row names. A pure config value; no code path changes.
INCIDENT_ESTIMATE_FLOOR = 16_000
#: The measured prompt of the incident's large-append turn (LP-0MW0KZUUS002LICZ).
INCIDENT_PROMPT_TOKENS = 19_628
#: The appended tool-role message that broke the estimator.
INCIDENT_APPEND_BYTES = 51_945
#: The measured resident prefix the pre-dispatch price trusted.
INCIDENT_RESIDENT_TOKENS = 6_622

#: Record kinds the window fixture may carry.
WINDOW_KINDS = ("measured_cold", "abandoned", "predicted_warm")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a committed fixture as ``(meta, records)``.

    Every record must name its kind and carry a non-empty provenance string
    (AC4): a row without one is fixture corruption, not a replayed event.
    """
    fixture = Path(path)
    with fixture.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise AssertionError(f"fixture {fixture} is empty")
    header = json.loads(lines[0])
    if "_meta" not in header:
        raise AssertionError(f"fixture {fixture} is missing its `_meta` header")
    records = [json.loads(line) for line in lines[1:]]
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, str) or not provenance.strip():
            raise AssertionError(f"fixture {fixture} row without provenance: {record}")
    return header["_meta"], records


def load_window(path: Path | None = None) -> tuple[dict[str, Any], list[dict]]:
    """The certified-window fixture: ``(meta, records)``."""
    return load_fixture(WINDOW_FIXTURE if path is None else path)


def load_incident(path: Path | None = None) -> tuple[dict[str, Any], list[dict]]:
    """The reconstructed 2026-11-03 sequence: ``(meta, records)``."""
    return load_fixture(INCIDENT_FIXTURE if path is None else path)


def _kind_rows(kind: str, records: list[dict] | None) -> list[dict]:
    """Records of *kind*, defaulting to the certified-window fixture."""
    resolved = load_window()[1] if records is None else records
    return [record for record in resolved if record.get("kind") == kind]


def measured_cold_rows(records: list[dict] | None = None) -> list[dict]:
    """The window's measured cold completions (the AC1 rate oracle)."""
    return _kind_rows("measured_cold", records)


def abandoned_rows(records: list[dict] | None = None) -> list[dict]:
    """The window's abandoned prefills (the cold-window sample source)."""
    return _kind_rows("abandoned", records)


def predicted_warm_rows(records: list[dict] | None = None) -> list[dict]:
    """The window's predicted-warm events (the AC2 refusal population)."""
    return _kind_rows("predicted_warm", records)


def rate_band(meta: dict[str, Any] | None = None) -> dict[str, float]:
    """The measured cold-rate band the fixture header carries (ms/token)."""
    resolved = load_window()[0] if meta is None else meta
    band = resolved["cold_rate_band_ms_per_token"]
    return {
        "min_ms_per_token": float(band["min"]),
        "max_ms_per_token": float(band["max"]),
        "min_tokens_per_second": 1000.0 / float(band["max"]),
        "max_tokens_per_second": 1000.0 / float(band["min"]),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def shipped_config(**overrides: Any) -> dict[str, Any]:
    """The shipped ``proxy/config.yaml``, with *overrides* on the server block.

    The replay runs against the settings the proxy actually ships, so the bound
    it proves is the bound production enforces. Overrides are used only for the
    two replay knobs documented on the item: the incident's pricing floor and a
    deliberately tighter deadline (the AC3 refusal branch).
    """
    data = yaml.safe_load(SHIPPED_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    server = dict(data.get("server") or {})
    server.update(overrides)
    return {"server": server}


def incident_config(**overrides: Any) -> dict[str, Any]:
    """Shipped config with the incident row's pricing floor applied."""
    server: dict[str, Any] = {
        "local_admission_guard_eta_percentile_min_tokens": INCIDENT_ESTIMATE_FLOOR
    }
    server.update(overrides)
    return shipped_config(**server)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def reset() -> None:
    """Clear every global the replay reads (calibration + session pricing).

    AC4: the harness resets calibration per test, so an earlier test's samples
    can never price a later replay. The provider's session state (recorded
    ratio / prompt prefix) is cleared for the same reason.
    """
    prefill_eta.reset()
    provider.clear_cached_ratio_for_session(SESSION)


# ---------------------------------------------------------------------------
# AC1 — the calibrated cold rate
# ---------------------------------------------------------------------------


def load_cold_window(records: list[dict] | None = None) -> list[dict]:
    """Feed the window's abandoned prefills through the real cold recorder.

    Returns the rows the recorder actually **accepted**: the exact population
    the fitted rate derives from. A rejected row would silently change the fit,
    so the count is part of the AC1 assertion rather than an implementation
    detail.
    """
    accepted: list[dict] = []
    for row in abandoned_rows(records):
        if prefill_eta.record_abandoned_prefill_sample(
            MODEL,
            row["total"],
            row["cache"],
            row["processed"],
            row["time_ms"],
        ):
            accepted.append(row)
    return accepted


def cold_rate(
    prompt_tokens: int, config: dict | None = None
) -> prefill_eta.DeadlineCapacity:
    """The calibrated cold capacity/rate the admission path prices with."""
    return prefill_eta.deadline_capacity(
        MODEL, prompt_tokens, shipped_config() if config is None else config
    )


# ---------------------------------------------------------------------------
# AC2 — replay the predicted-warm events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayOutcome:
    """One replayed window event and the real decision it produced."""

    row: dict[str, Any]
    decision: prefill_eta.AdmissionDecision

    @property
    def admitted(self) -> bool:
        return bool(self.decision.admit)

    @property
    def capacity_tokens(self) -> int | None:
        return self.decision.deadline_capacity_tokens

    @property
    def priced_tokens(self) -> int:
        return self.decision.capacity_priced_tokens

    @property
    def cold_projection_seconds(self) -> float:
        """The row's cold projection: its priced tokens at the decision's rate.

        This is the quantity the capacity expresses in tokens, restated in
        seconds, so "no admitted request had a cold projection exceeding the
        capacity bound" can be asserted in both units.
        """
        rate = self.decision.cold_ms_per_token
        if not rate:
            raise AssertionError(f"decision carries no cold rate: {self.row}")
        return self.priced_tokens * float(rate) / 1000.0


def replay_predicted_warm(
    config: dict | None = None, records: list[dict] | None = None
) -> list[ReplayOutcome]:
    """Replay every predicted-warm event through the real admission decision.

    Each row is replayed on the honest residency basis: the admission *predicted*
    a reused prefix, but a prediction is not evidence, so the capacity prices the
    whole prompt cold (``residency_measured=False``). The row's own predicted
    ratio is still the admission ratio, exactly as the shipped path priced it.
    """
    resolved = shipped_config() if config is None else config
    outcomes: list[ReplayOutcome] = []
    for row in predicted_warm_rows(records):
        total = int(row["total"])
        predicted_ratio = 1.0 - float(row["priced_new_tokens"]) / total
        decision = prefill_eta.evaluate_admission(
            resolved,
            MODEL,
            total,
            predicted_ratio,
            "low",
            residency_measured=False,
        )
        outcomes.append(ReplayOutcome(row=row, decision=decision))
    return outcomes


# ---------------------------------------------------------------------------
# AC3 — the 2026-11-03 incident
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncidentPricing:
    """The pre-dispatch price of the incident's appended turn."""

    row: dict[str, Any]
    priced_cached_ratio: float
    resident_tokens: int
    cold_ms_per_token: float
    capacity_tokens: int | None

    @property
    def forced_full_prefill(self) -> bool:
        """True when the shipped large-append rule zeroed the prefix credit."""
        return self.priced_cached_ratio == 0.0

    @property
    def cold_price_seconds(self) -> float:
        """The whole-prompt cold price the forced ratio produces."""
        return (
            int(self.row["prompt_tokens"]) * self.cold_ms_per_token / 1000.0
        )


@dataclass(frozen=True)
class IncidentOutcome:
    """One replayed incident turn: its price and the decision it produced."""

    pricing: IncidentPricing
    decision: prefill_eta.AdmissionDecision

    @property
    def admitted(self) -> bool:
        return bool(self.decision.admit)

    @property
    def cold_price_seconds(self) -> float:
        return self.pricing.cold_price_seconds

    @property
    def capacity_tokens(self) -> int | None:
        return self.decision.deadline_capacity_tokens

    @property
    def priced_tokens(self) -> int:
        return self.decision.capacity_priced_tokens

    @property
    def over_capacity(self) -> bool:
        """Whether the priced prompt exceeds the decision's capacity bound."""
        capacity = self.capacity_tokens
        return capacity is not None and self.priced_tokens > capacity


def incident_base_messages() -> list[dict[str, Any]]:
    """The already-dispatched prefix the session's recorded prompt covers."""
    return [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "read the skill file"},
    ]


def incident_messages(
    append_bytes: int = INCIDENT_APPEND_BYTES,
) -> list[dict[str, Any]]:
    """The incident's incoming prompt: the recorded prefix + the tool append.

    The append is a single tool-role message whose ``role + content`` occupies
    *append_bytes* (51,945 bytes in the incident — a full SKILL.md pasted into
    history).
    """
    overhead = len("tool") + 1  # role + the fingerprint separator
    if append_bytes <= overhead:
        raise ValueError("append_bytes must exceed the message overhead")
    return incident_base_messages() + [
        {"role": "tool", "content": "x" * (append_bytes - overhead)}
    ]


def _prime_incident_session() -> None:
    """Restore the pre-dispatch session state the incident recorded.

    ``set_cached_ratio`` 0.99 is the measured ratio of the previous local
    response; the recorded prompt prefix is the 6,622-token prefix llama-server
    actually held. The large-append rule compares the incoming content against
    that prefix.
    """
    provider.clear_cached_ratio_for_session(SESSION)
    provider.set_cached_ratio(MODEL, SESSION, 0.99)
    provider.record_prompt_prefix(
        MODEL, SESSION, INCIDENT_RESIDENT_TOKENS, incident_base_messages()
    )


def price_incident_turn(config: dict | None = None) -> IncidentPricing:
    """Price the incident's appended turn through the shipped large-append rule."""
    resolved = incident_config() if config is None else config
    row = _kind_rows("incident_turn", load_incident()[1])[0]
    _prime_incident_session()
    ratio = provider.expected_cached_ratio(
        MODEL,
        SESSION,
        int(row["prompt_tokens"]),
        incident_messages(),
        server_config=resolved.get("server", resolved),
    )
    capacity = prefill_eta.deadline_capacity(
        MODEL, int(row["prompt_tokens"]), resolved
    )
    rate = capacity.cold_ms_per_token
    if rate is None:
        raise AssertionError("the incident replay has no calibrated cold rate")
    return IncidentPricing(
        row=row,
        priced_cached_ratio=float(ratio),
        resident_tokens=INCIDENT_RESIDENT_TOKENS,
        cold_ms_per_token=float(rate),
        capacity_tokens=capacity.tokens,
    )


def replay_incident(
    config: dict | None = None,
    *,
    deadline_seconds: float | None = None,
    admission_ratio: float | None = None,
) -> IncidentOutcome:
    """Replay the incident's appended turn: its cold price and its verdict.

    ``deadline_seconds`` replaces the shipped ``local_prefill_deadline_seconds``
    (the tighter-deadline refusal branch); ``admission_ratio`` replaces the
    forced-cold ratio with the measured one, which is how the bound is isolated
    from the temporal deadline when both fire on the same cold price.

    The replay always prices the **measured** prompt (19,628 tokens) the incident
    row records, not the pre-dispatch estimate (18,516): that is the work
    llama-server actually did, and it is the number the item names.
    """
    resolved = incident_config() if config is None else config
    if deadline_seconds is not None:
        server = dict(resolved.get("server", resolved))
        server["local_prefill_deadline_seconds"] = deadline_seconds
        resolved = {"server": server}
    pricing = price_incident_turn(resolved)
    ratio = (
        pricing.priced_cached_ratio
        if admission_ratio is None
        else float(admission_ratio)
    )
    decision = prefill_eta.evaluate_admission(
        resolved,
        MODEL,
        int(pricing.row["prompt_tokens"]),
        ratio,
        "low",
        residency_measured=False,
    )
    return IncidentOutcome(pricing=pricing, decision=decision)


def incident_turns() -> list[dict[str, Any]]:
    """The reconstructed incident's recorded turns, in sequence order."""
    return _kind_rows("incident_turn", load_incident()[1])


def incident_measured_ratio(pricing: IncidentPricing) -> float:
    """The measured cached ratio of the incident row (6,622 of 19,628 resident)."""
    row = pricing.row
    return (int(row["prompt_tokens"]) - int(row["new_tokens"])) / int(
        row["prompt_tokens"]
    )
