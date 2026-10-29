"""Offline tests for the slot snapshot save/restore benchmark harness.

Covers (LP-0MVOG8SYO006R49J):

- band prompt sizing for the 10k / 25k / 40k / 60k+ bands;
- the adaptive-timeout model ``min(3.0 + 0.0015 * tokens, 60)`` including the
  60 s cap boundary;
- the result-record schema (save wall time, restore wall time, bytes,
  outcome, ``cached_tokens``);
- that a failed save/restore is **recorded with its error and never retried
  away**;
- the "under load" mode (saves issue while another slot is processing),
  proven with a stubbed HTTP client;
- the token-based hit definition ``cached_tokens = prompt_total -
  timings.prompt_n``;
- the snapshot-byte cross-check against ``kv_memory_table.py``; and
- the AC5 guard: no production module imports the harness.

No GPU and no live llama-server/proxy are required: every test drives the
module through a stubbed HTTP client or pure functions.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest

try:
    from benchmarks import kv_memory_table as kvt
    from benchmarks import slot_snapshot_benchmark as m
except ImportError:
    try:
        from proxy.benchmarks import kv_memory_table as kvt
        from proxy.benchmarks import slot_snapshot_benchmark as m
    except ImportError:
        this_dir = Path(__file__).resolve().parent
        proxy_dir = this_dir.parent  # proxy/
        root_dir = proxy_dir.parent  # project root
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from proxy.benchmarks import kv_memory_table as kvt
        from proxy.benchmarks import slot_snapshot_benchmark as m

MODEL = "Qwen3"
TIMEOUT_MODEL = m.TimeoutModel(
    base_seconds=3.0, per_token_seconds=0.0015, cap_seconds=60.0
)


# ---------------------------------------------------------------------------
# Stubbed HTTP client
# ---------------------------------------------------------------------------


class StubStream:
    """A streaming completion whose read blocks until the test closes it."""

    def __init__(self, client: StubClient) -> None:
        self._client = client
        self._opened = asyncio.Event()
        self._closed = asyncio.Event()
        self._closed_now = False

    async def start(self) -> None:
        self._opened.set()
        self._client.stream_open = True
        self._client.events.append(("stream_started", None))

    async def read(self):
        # Block until close() — the stream stays "processing" the whole window,
        # so the harness's save/restore calls necessarily happen while another
        # slot is busy.
        await self._closed.wait()
        return None

    async def close(self) -> None:
        if self._closed_now:
            return
        self._closed_now = True
        self._closed.set()
        self._client.stream_open = False
        self._client.events.append(("stream_closed", None))


class StubClient:
    """Recording HTTP client; failures/interleaving are asserted by the tests."""

    def __init__(
        self,
        save_status: int = 200,
        restore_status: int = 200,
        save_error: BaseException | None = None,
        restore_error: BaseException | None = None,
        verification: dict | None = None,
        populate: dict | None = None,
        populate_status: int = 200,
        populate_error: BaseException | None = None,
        save_body: dict | None = None,
        restore_body: dict | None = None,
        erase_status: int = 200,
        populate_failures: int = 0,
    ) -> None:
        self.save_status = save_status
        self.restore_status = restore_status
        self.save_error = save_error
        self.restore_error = restore_error
        self.verification = verification
        self.save_body = save_body
        self.restore_body = restore_body
        self.erase_status = erase_status
        self.populate = populate
        self.populate_status = populate_status
        self.populate_error = populate_error
        # How many populate attempts to answer with a retryable 5xx before
        # serving the real populate body.
        self.populate_failures = populate_failures
        self.populate_expected = (
            populate is not None
            or populate_error is not None
            or populate_status != 200
            or populate_failures > 0
        )
        self.populate_served = False
        self.calls: list[dict] = []  # every post/open_stream
        self.events: list[tuple[str, object]] = []
        self.stream_open = False

    async def post(self, url: str, payload: dict, timeout: float) -> m.SlotResponse:
        assert (
            "action=save" in url
            or "action=restore" in url
            or "action=erase" in url
            or "completions" in url
        )
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout": timeout,
                "stream_open": self.stream_open,
            }
        )
        self.events.append(("post", url))
        if "action=save" in url:
            if self.save_error is not None:
                raise self.save_error
            return m.SlotResponse(
                status_code=self.save_status, body=dict(self.save_body or {"ok": True})
            )
        if "action=restore" in url:
            if self.restore_error is not None:
                raise self.restore_error
            return m.SlotResponse(
                status_code=self.restore_status,
                body=dict(self.restore_body or {"ok": True}),
            )
        if "action=erase" in url:
            return m.SlotResponse(
                status_code=self.erase_status, body={"id_slot": 1, "n_erased": 0}
            )
        # Chat completions: the populate probe precedes the save, the
        # verification probe follows the restore.
        if self.populate_expected and not self.populate_served:
            self.populate_calls = getattr(self, "populate_calls", 0) + 1
            if self.populate_calls <= self.populate_failures:
                # A retryable administrative failure (5xx): the instance was
                # recycled mid-prefill, so the populate is re-issued.
                return m.SlotResponse(
                    status_code=500,
                    body="proxy error: Failed to read connection",
                )
            self.populate_served = True
            if self.populate_error is not None:
                raise self.populate_error
            return m.SlotResponse(
                status_code=self.populate_status,
                body=dict(self.populate) if self.populate is not None else {"choices": []},
            )
        if self.verification is not None:
            return m.SlotResponse(status_code=200, body=dict(self.verification))
        return m.SlotResponse(status_code=200, body={"choices": []})

    def open_stream(self, url: str, payload: dict, timeout: float) -> StubStream:
        self.calls.append(
            {"url": url, "payload": payload, "timeout": timeout, "stream": True}
        )
        return StubStream(self)

    def save_calls(self) -> list[dict]:
        return [c for c in self.calls if "action=save" in c["url"]]

    def restore_calls(self) -> list[dict]:
        return [c for c in self.calls if "action=restore" in c["url"]]

    def completion_calls(self) -> list[dict]:
        return [c for c in self.calls if "completions" in c["url"] and "stream" not in c]


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    return tmp_path / "slot-cache"


BAND_10K = m.BandSpec("10k", 10_000)
BAND_25K = m.BandSpec("25k", 25_000)
BAND_40K = m.BandSpec("40k", 40_000)
BAND_60K = m.BandSpec("60k+", 60_000)


# ---------------------------------------------------------------------------
# Bands and prompt sizing (AC: band prompt sizing for 10k/25k/40k/60k+)
# ---------------------------------------------------------------------------


def test_default_bands_cover_the_four_required_ranges():
    bands = m.DEFAULT_BANDS
    assert [b.name for b in bands] == ["10k", "25k", "40k", "60k+"]
    assert [b.target_tokens for b in bands] == [10_000, 25_000, 40_000, 60_000]


def test_parse_bands_accepts_names_and_token_counts():
    assert [b.name for b in m.parse_bands("10k,25k,40k,60k+")] == [
        "10k",
        "25k",
        "40k",
        "60k+",
    ]
    assert [b.target_tokens for b in m.parse_bands("10000,60000")] == [10_000, 60_000]
    assert m.parse_bands("") == m.DEFAULT_BANDS
    assert m.parse_bands(None) == m.DEFAULT_BANDS
    with pytest.raises(ValueError):
        m.parse_bands("bogus")
    with pytest.raises(ValueError):
        m.parse_bands("10k,10k")


def test_every_band_prompt_stays_inside_its_band():
    """Sizing is deterministic and lands *in* the band, never past it.

    Overshooting is not harmless: the first measurement run inherited a 4.0
    chars/token assumption and turned the 10k band into a 40k prompt, which
    moved the measurement out of the band it was supposed to describe.
    """
    for band in m.DEFAULT_BANDS:
        for salt in ("salt-a", "salt-b"):
            prompt = m.band_prompt(band, salt)
            estimated = m.estimated_prompt_tokens(prompt)
            assert band.target_tokens <= estimated <= band.target_tokens * 1.01, (
                f"{band.name} prompt estimates {estimated} tokens, outside "
                f"[{band.target_tokens}, {band.target_tokens * 1.01:.0f}]"
            )
            assert m.build_salted_prompt(band.target_tokens, salt) == prompt, (
                "same salt must build the same prompt (determinism)"
            )


def test_band_prompts_are_salted_and_monotonic():
    """Different bands must not share a KV prefix; sizes must order correctly."""
    salt = "bench-run-1"
    prompts = {
        band.name: m.band_prompt(band, salt) for band in m.DEFAULT_BANDS
    }
    # The salt is the first thing in the prompt, so prefixes never collide
    # across bands — otherwise a shorter snapshot would serve a longer restore.
    seen_prefixes = {p.split("\n", 1)[0] for p in prompts.values()}
    assert len(seen_prefixes) == len(prompts)
    # Different salts produce different content, not just a different header.
    a = m.band_prompt(m.BandSpec("10k", 10_000), "salt-x")
    b = m.band_prompt(m.BandSpec("10k", 10_000), "salt-y")
    header_len = a.index("\n") + 1
    assert a[header_len : header_len + 100] != b[header_len : header_len + 100]
    # Larger band -> strictly more estimated tokens.
    sizes = [m.estimated_prompt_tokens(p) for p in prompts.values()]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_band_prompt_rejects_non_positive_targets():
    with pytest.raises(ValueError):
        m.build_salted_prompt(0, "x")
    with pytest.raises(ValueError):
        m.build_salted_prompt(-5, "x")


# ---------------------------------------------------------------------------
# Adaptive timeout model (AC: min(3.0 + 0.0015 * tokens, 60), cap boundary)
# ---------------------------------------------------------------------------


def test_adaptive_timeout_matches_the_production_formula():
    """The shipped model is min(3.0 + 0.0015*tokens, 60) at three bands."""
    assert TIMEOUT_MODEL.timeout_for(10_000) == pytest.approx(18.0)
    assert TIMEOUT_MODEL.timeout_for(25_000) == pytest.approx(40.5)
    assert TIMEOUT_MODEL.timeout_for(40_000) == pytest.approx(60.0)  # capped
    assert TIMEOUT_MODEL.timeout_for(100_000) == pytest.approx(60.0)


def test_adaptive_timeout_cap_boundary_is_exact():
    """The 60 s cap engages exactly at (cap - base) / per_token = 38000."""
    cap_tokens = TIMEOUT_MODEL.cap_tokens
    assert cap_tokens == pytest.approx(38_000.0)
    below = int(cap_tokens) - 1
    assert TIMEOUT_MODEL.timeout_for(below) < 60.0
    assert TIMEOUT_MODEL.timeout_for(below + 1) == pytest.approx(60.0)
    # Just below the cap the model is still strictly linear.
    assert TIMEOUT_MODEL.timeout_for(38_000) == pytest.approx(60.0)


def test_timeout_model_from_server_config_overrides_defaults():
    config = {
        "session_slot_timeout_seconds": 2,
        "session_slot_timeout_per_token_seconds": 0.001,
        "session_slot_max_timeout_seconds": 45,
    }
    model = m.timeout_model_from_config(config)
    assert model.timeout_for(10_000) == pytest.approx(12.0)
    assert model.timeout_for(100_000) == pytest.approx(45.0)
    # A config missing a key falls back to that key's default only.
    partial = {"session_slot_timeout_per_token_seconds": 0.0015}
    model = m.timeout_model_from_config(partial)
    assert model.base_seconds == 3.0
    assert model.cap_seconds == 60.0
    # Garbage values fail open to defaults.
    model = m.timeout_model_from_config({"session_slot_max_timeout_seconds": "oops"})
    assert model.cap_seconds == 60.0


# ---------------------------------------------------------------------------
# Result-record schema (AC3)
# ---------------------------------------------------------------------------


def test_record_to_json_has_exactly_the_documented_schema():
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="session-a",
            salt="s1",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    payload = record.to_json()
    assert set(payload) == set(m.RECORD_FIELDS)
    assert payload["band"] == "10k"
    assert payload["target_tokens"] == 10_000
    assert payload["save_outcome"] == m.OUTCOME_SUCCESS
    assert payload["restore_outcome"] == m.OUTCOME_SUCCESS
    assert payload["save_seconds"] is not None and payload["save_seconds"] >= 0
    assert payload["restore_seconds"] is not None
    # JSON-roundtrips cleanly (the CLI/API contract).
    assert json.loads(json.dumps(payload)) == payload


def test_record_carries_prompt_tokens_from_verification_when_enabled():
    """--verify-chat records usage.prompt_tokens and timings.prompt_n."""
    stub = StubClient(
        verification={
            "usage": {"prompt_tokens": 25_037},
            "timings": {"prompt_n": 537},
        }
    )
    record = _run(
        m.measure_band(
            BAND_25K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="session-verify",
            salt="sv",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            verify_chat=True,
        )
    )
    assert record.verified is True
    assert record.prompt_total_tokens == 25_037
    assert record.prompt_processed_tokens == 537
    assert record.cached_tokens == 25_037 - 537
    assert record.cached_ratio == pytest.approx(1 - 537 / 25_037)


def test_record_falls_back_to_the_estimate_without_verification():
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.verified is False
    assert record.cached_tokens is None
    assert record.prompt_total_tokens == record.prompt_tokens_estimated
    assert record.prompt_tokens_estimated >= BAND_10K.target_tokens
    # The optional populate step is off unless asked for: no completion ran.
    assert record.populate_outcome == m.OUTCOME_NOT_REQUESTED
    assert stub.completion_calls() == []


# ---------------------------------------------------------------------------
# Slot populate and slot pinning (what makes a band measurable at all)
# ---------------------------------------------------------------------------


def test_populate_prefills_the_measured_slot_before_the_save():
    """The populate probe runs first, pinned, so the save captures real KV."""
    stub = StubClient(
        populate={
            "usage": {"prompt_tokens": 25_100},
            "timings": {"prompt_n": 25_100},
        },
        verification={
            "usage": {"prompt_tokens": 25_037},
            "timings": {"prompt_n": 37},
        },
    )
    record = _run(
        m.measure_band(
            BAND_25K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
            verify_chat=True,
            verify_slot_id=1,
        )
    )
    # Ordering: erase → populate → save → restore → verification. The erase
    # clears the previous band's KV so this band's snapshot is not coupled to it.
    assert "action=erase" in stub.calls[0]["url"]
    assert "completions" in stub.calls[1]["url"]
    assert "action=save" in stub.calls[2]["url"]
    assert "action=restore" in stub.calls[3]["url"]
    assert record.erase_outcome == m.OUTCOME_SUCCESS
    assert len(stub.completion_calls()) == 2
    # The populate is pinned to the measured slot and is not bound by the
    # adaptive save/restore timeout (a 60k prefill would exceed it).
    populate_call = stub.completion_calls()[0]
    assert populate_call["payload"]["id_slot"] == 1
    assert populate_call["timeout"] == m.POPULATE_TIMEOUT_SECONDS
    assert m.POPULATE_TIMEOUT_SECONDS > TIMEOUT_MODEL.cap_seconds

    assert record.populate_outcome == m.OUTCOME_SUCCESS
    assert record.populate_seconds is not None
    assert record.populate_prompt_total_tokens == 25_100
    # The verification completion's prompt size wins (it describes the
    # restored slot); the populate size is kept as recorded context.
    assert record.prompt_total_tokens == 25_037
    assert record.cached_tokens == 25_037 - 37


def test_populate_total_sizes_the_record_when_verification_is_off():
    stub = StubClient(
        populate={
            "usage": {"prompt_tokens": 40_512},
            "timings": {"prompt_n": 40_512},
        }
    )
    record = _run(
        m.measure_band(
            BAND_40K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.verified is False
    assert record.cached_tokens is None
    assert record.prompt_total_tokens == 40_512
    assert record.expected_snapshot_bytes == m.expected_snapshot_bytes(40_512)


def test_deterministic_populate_failure_is_recorded_and_not_retried():
    """A 4xx will fail identically, so it is reported once, not re-issued."""
    stub = StubClient(populate_status=400)
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.populate_outcome == m.OUTCOME_FAILED
    assert record.populate_error is not None and "400" in record.populate_error
    assert record.populate_attempts == 1, "a 4xx populate is deterministic"
    # The save/restore still measure (a failed populate only weakens the claim).
    assert record.save_outcome == m.OUTCOME_SUCCESS
    assert record.restore_outcome == m.OUTCOME_SUCCESS
    assert record.populate_prompt_total_tokens is None


def test_a_recycled_instance_populate_failure_is_retried():
    """An administrative 5xx (instance recycled mid-prefill) is re-issued.

    Observed on the production host: a 40k populate ended in
    ``HTTP 500: proxy error: Failed to read connection`` after 600 s because
    the proxy's model-health loop reloaded the instance during the prefill.
    Retrying the *administrative* populate cannot flatter the save/restore
    numbers it prepares — it only stops an unrelated reload from discarding
    the band.
    """
    stub = StubClient(
        populate_failures=2,
        populate={"usage": {"prompt_tokens": 10_010}, "timings": {"prompt_n": 10_010}},
    )
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.populate_outcome == m.OUTCOME_SUCCESS
    assert record.populate_attempts == 3
    # Each attempt re-erases, so the retried populate still starts from a
    # cleared slot rather than layering onto a half-written prefix.
    assert sum(1 for c in stub.calls if "action=erase" in c["url"]) == 3
    assert record.populate_prompt_total_tokens == 10_010
    assert record.save_outcome == m.OUTCOME_SUCCESS


def test_populate_retries_are_bounded():
    stub = StubClient(populate_failures=m.POPULATE_ATTEMPTS + 1)
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.populate_outcome == m.OUTCOME_FAILED
    assert record.populate_attempts == m.POPULATE_ATTEMPTS
    assert len(stub.completion_calls()) == m.POPULATE_ATTEMPTS


def test_slot_actions_carry_the_model_for_the_router_path():
    """Save/restore/erase must reach the instance behind the router.

    The proxy issues saves through ``localhost:{llama_server_port}/slots/..``
    with a ``model`` body field (``proxy/proxy/session.py:_call_slot_endpoint``),
    because the llama-server router refuses an action without a model name.
    Measuring through the router is the only port-stable path on the
    production host: instance ports churn as the router recycles them, while
    the router port does not.
    """
    stub = StubClient()
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    slot_calls = [
        c for c in stub.calls if "action=" in c["url"]
    ]
    assert slot_calls, "expected erase/save/restore calls"
    for call in slot_calls:
        assert call["payload"].get("model") == MODEL, call["url"]


def test_verify_slot_id_pins_the_verification_completion():
    stub = StubClient(verification={"usage": {"prompt_tokens": 10_050}})
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=2,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            verify_chat=True,
            verify_slot_id=2,
        )
    )
    verify_call = stub.completion_calls()[-1]
    assert verify_call["payload"]["id_slot"] == 2


def test_verification_is_unpinned_by_default():
    stub = StubClient(verification={"usage": {"prompt_tokens": 10_050}})
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=2,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            verify_chat=True,
        )
    )
    assert "id_slot" not in stub.completion_calls()[-1]["payload"]


def test_load_spec_pins_the_stream_to_another_slot():
    """`--under-load` must stream on the *other* slot, not the measured one."""
    stub = StubClient()
    load = m.LoadSpec(model=MODEL, max_tokens=64, id_slot=0)
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            load=load,
        )
    )
    stream_call = [c for c in stub.calls if c.get("stream")][0]
    assert stream_call["payload"]["id_slot"] == 0
    assert load.payload()["stream"] is True
    # Unpinned by default.
    assert "id_slot" not in m.LoadSpec(model=MODEL).payload()


# ---------------------------------------------------------------------------
# Server-reported sizes / timings, and per-band slot isolation
# ---------------------------------------------------------------------------


def test_save_prefers_server_reported_bytes_and_keeps_the_disk_stat(tmp_path):
    """`n_written` is authoritative; the on-disk stat is kept as corroboration.

    Statting alone raced a late write in the first live run and reported a
    previous band's file as this band's footprint.
    """
    snapshot_dir = tmp_path / "slot-cache"
    snapshot_dir.mkdir()
    # A stale file with the *next* band's deterministic name.
    stale = snapshot_dir / "snapshot-bench-10k-x.bin"
    stale.write_bytes(b"y" * 500)
    stub = StubClient(
        save_body={
            "id_slot": 1,
            "n_saved": 10_000,
            "n_written": 999_999,
            "timings": {"save_ms": 12.5},
        },
        restore_body={
            "id_slot": 1,
            "n_restored": 10_000,
            "n_read": 999_999,
            "timings": {"restore_ms": 3.5},
        },
    )
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            snapshot_dir=snapshot_dir,
        )
    )
    assert record.snapshot_bytes == 999_999
    assert record.snapshot_bytes_on_disk == 500
    assert record.save_server_ms == pytest.approx(12.5)
    assert record.save_tokens == 10_000
    assert record.restore_server_ms == pytest.approx(3.5)
    assert record.restore_tokens == 10_000


def test_snapshot_is_flagged_when_it_does_not_hold_the_band():
    """A save of the wrong KV is a successful call but not a band measurement.

    The first full run reported a 40k under-load band as ``success`` while
    ``n_saved`` was 3 878 — the slot had been replaced between the populate and
    the save (a model reload). Nothing in the record said so; only the absurd
    0.25 snapshot-size ratio hinted at it.
    """
    stub = StubClient(
        populate={"usage": {"prompt_tokens": 39_942}, "timings": {"prompt_n": 39_942}},
        save_body={"id_slot": 1, "n_saved": 3_878, "n_written": 108_150_404},
    )
    record = _run(
        m.measure_band(
            BAND_40K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.save_outcome == m.OUTCOME_SUCCESS
    assert record.save_tokens == 3_878
    assert record.snapshot_holds_band is False


def test_snapshot_holding_the_band_is_accepted_within_tolerance():
    stub = StubClient(
        populate={"usage": {"prompt_tokens": 39_942}, "timings": {"prompt_n": 39_942}},
        save_body={"id_slot": 1, "n_saved": 39_949, "n_written": 719_548_588},
    )
    record = _run(
        m.measure_band(
            BAND_40K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.snapshot_holds_band is True


def test_snapshot_fidelity_is_unknown_without_a_populate_or_server_count():
    record = _run(
        m.measure_band(
            BAND_10K,
            client=StubClient(),
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.snapshot_holds_band is None


def test_an_empty_slot_at_save_time_is_a_failed_fidelity_check():
    """``n_saved: 0`` means the slot was empty — a failure, not missing data.

    Observed on a loaded host: the erase timed out client-side at 120 s but
    still executed server-side after the populate, so the save that followed
    captured an empty slot while reporting ``success`` and 756 bytes.
    """
    stub = StubClient(
        populate={"usage": {"prompt_tokens": 59_942}, "timings": {"prompt_n": 59_942}},
        save_body={"id_slot": 1, "n_saved": 0, "n_written": 756},
        restore_body={"id_slot": 1, "n_restored": 0},
    )
    record = _run(
        m.measure_band(
            m.DEFAULT_BANDS[-1],
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.save_outcome == m.OUTCOME_SUCCESS
    assert record.save_tokens == 0
    assert record.snapshot_holds_band is False
    assert m._snapshot_holds_band(0, 59_942) is False


def test_server_timings_are_null_when_the_server_does_not_report_them():
    record = _run(
        m.measure_band(
            BAND_10K,
            client=StubClient(),
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.save_server_ms is None
    assert record.restore_server_ms is None
    assert record.save_tokens is None


def test_erase_isolates_the_band_before_it_populates():
    stub = StubClient(
        populate={"usage": {"prompt_tokens": 10_010}, "timings": {"prompt_n": 10_010}}
    )
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    # erase → populate → save → restore
    assert "action=erase" in stub.calls[0]["url"]
    assert "completions" in stub.calls[1]["url"]
    assert "action=save" in stub.calls[2]["url"]
    assert record.erase_outcome == m.OUTCOME_SUCCESS
    assert record.erase_seconds is not None


def test_erase_and_populate_are_not_bound_by_the_adaptive_timeout():
    """Administrative steps get their own budget, not the measured band's.

    The 10k band's adaptive budget is 18 s, which the first live run spent
    queueing for an erase the host answers in ~14 ms when idle — the erase was
    recorded as ``failed`` purely because it was issued with the measurement
    budget. Both preparatory calls must therefore use their own generous bound.
    """
    stub = StubClient(
        populate={"usage": {"prompt_tokens": 10_010}, "timings": {"prompt_n": 10_010}}
    )
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=1,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    erase_call = [c for c in stub.calls if "action=erase" in c["url"]][0]
    populate_call = stub.completion_calls()[0]
    assert erase_call["timeout"] == m.ERASE_TIMEOUT_SECONDS
    assert populate_call["timeout"] == m.POPULATE_TIMEOUT_SECONDS
    # Both exceed the band's own budget, so neither can be mistaken for a
    # save/restore timeout (the thing this benchmark exists to measure).
    assert m.ERASE_TIMEOUT_SECONDS > TIMEOUT_MODEL.cap_seconds
    assert m.POPULATE_TIMEOUT_SECONDS > TIMEOUT_MODEL.cap_seconds
    band_budget = TIMEOUT_MODEL.timeout_for(BAND_10K.target_tokens)
    assert m.ERASE_TIMEOUT_SECONDS > band_budget


def test_erase_is_not_issued_without_populate():
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.erase_outcome == m.OUTCOME_NOT_REQUESTED
    assert not any("action=erase" in call["url"] for call in stub.calls)


def test_failed_erase_is_recorded_and_the_measurement_proceeds():
    stub = StubClient(
        erase_status=500,
        populate={"usage": {"prompt_tokens": 10_010}, "timings": {"prompt_n": 10_010}},
    )
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            populate_slot=True,
        )
    )
    assert record.erase_outcome == m.OUTCOME_FAILED
    assert record.populate_outcome == m.OUTCOME_SUCCESS
    assert record.save_outcome == m.OUTCOME_SUCCESS



# ---------------------------------------------------------------------------
# Failure policy (AC: recorded with its error, never retried away)
# ---------------------------------------------------------------------------


def test_failed_save_is_recorded_with_error_and_never_retried():
    stub = StubClient(save_status=503)
    record = _run(
        m.measure_band(
            BAND_25K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    saves = stub.save_calls()
    assert len(saves) == 1, "a failed save must never be retried"
    assert record.save_outcome == m.OUTCOME_FAILED
    assert record.save_error is not None and "503" in record.save_error
    assert record.save_within_timeout is False
    # The restore is skipped, not issued against a snapshot that was never
    # written.
    assert stub.restore_calls() == []
    assert record.restore_outcome == m.OUTCOME_SKIPPED
    assert "save failed" in record.restore_error
    assert record.snapshot_bytes is None


def test_save_network_exception_is_recorded_not_swallowed():
    stub = StubClient(save_error=RuntimeError("connection refused"))
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert len(stub.save_calls()) == 1
    assert record.save_outcome == m.OUTCOME_FAILED
    assert "RuntimeError" in record.save_error
    assert "connection refused" in record.save_error


def test_failed_restore_is_recorded_with_error_and_never_retried():
    stub = StubClient(restore_status=500)
    record = _run(
        m.measure_band(
            BAND_40K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.save_outcome == m.OUTCOME_SUCCESS
    restores = stub.restore_calls()
    assert len(restores) == 1, "a failed restore must never be retried"
    assert record.restore_outcome == m.OUTCOME_FAILED
    assert record.restore_error is not None and "500" in record.restore_error
    assert record.restore_within_timeout is False


def test_successful_save_records_elapsed_and_timeout_comparison():
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.save_outcome == m.OUTCOME_SUCCESS
    assert record.save_seconds is not None
    assert record.save_within_timeout is True
    # The timeout is the adaptive model evaluated at the band's *measured*
    # prompt estimate (which is at least the 10k floor).
    assert record.prompt_tokens_estimated >= 10_000
    assert record.timeout_seconds == pytest.approx(
        TIMEOUT_MODEL.timeout_for(record.prompt_tokens_estimated)
    )
    assert record.timeout_seconds == pytest.approx(
        3.0 + 0.0015 * record.prompt_tokens_estimated
    )


def test_snapshot_filename_is_sanitised_from_the_salt():
    """A hostile salt can never escape the snapshot directory."""
    stub = StubClient()
    _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="a/b:c d",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    save_filename = stub.save_calls()[0]["payload"]["filename"]
    restore_filename = stub.restore_calls()[0]["payload"]["filename"]
    assert save_filename == restore_filename
    assert "/" not in save_filename
    assert ":" not in save_filename
    assert " " not in save_filename
    assert save_filename.endswith(".bin")


# ---------------------------------------------------------------------------
# Under-load mode (AC1 clause; proven with a stubbed HTTP client)
# ---------------------------------------------------------------------------


def test_under_load_issues_saves_while_another_slot_is_streaming():
    """With a LoadSpec, the save/restore must land inside an open stream."""
    stub = StubClient()
    load = m.LoadSpec(model=MODEL, max_tokens=64)
    record = _run(
        m.measure_band(
            BAND_25K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            load=load,
        )
    )
    assert record.under_load is True
    events = list(stub.events)
    assert ("stream_started", None) in events
    assert ("stream_closed", None) in events
    # The save issued while the load stream was open (another slot busy).
    assert stub.save_calls()
    assert all(c["stream_open"] for c in stub.save_calls()), (
        "saves must issue while the load stream is open"
    )
    assert all(c["stream_open"] for c in stub.restore_calls())
    # Ordering: stream started before the first save; closed after the last
    # restore — the whole pair ran inside the busy window.
    save_posts = [
        i
        for i, (name, url) in enumerate(events)
        if name == "post" and isinstance(url, str) and "action=save" in url
    ]
    assert save_posts, "no save event recorded"
    started = events.index(("stream_started", None))
    closed = events.index(("stream_closed", None))
    assert started < save_posts[0] < closed


def test_without_load_stream_is_never_opened():
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="x",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
        )
    )
    assert record.under_load is False
    assert ("stream_started", None) not in stub.events


# ---------------------------------------------------------------------------
# Snapshot bytes cross-check against kv_memory_table.py
# ---------------------------------------------------------------------------


def test_expected_snapshot_bytes_uses_the_kv_memory_table_constant():
    """The size formula is the documented kv_memory_table per-token cost."""
    for tokens in (10_000, 25_000, 60_000):
        expected = int(kvt.KV_MIB_PER_TOKEN * tokens * 1024 * 1024)
        assert m.expected_snapshot_bytes(tokens) == expected
    # Sanity: 25k tokens should be on the order of a few hundred MiB.
    assert 200 * 1024 * 1024 < m.expected_snapshot_bytes(25_000) < 400 * 1024 * 1024


def test_measured_bytes_within_tolerance_asserted_offline():
    """snapshot_size_in_tolerance implements the documented ratio band."""
    tokens = 50_000
    expected = m.expected_snapshot_bytes(tokens)
    assert m.snapshot_size_in_tolerance(expected, tokens) is True
    assert m.snapshot_size_in_tolerance(expected // 2, tokens) is True
    assert m.snapshot_size_in_tolerance(expected * 2, tokens) is True
    assert m.snapshot_size_in_tolerance(expected // 10, tokens) is False
    assert m.snapshot_size_in_tolerance(expected * 10, tokens) is False
    assert m.snapshot_size_in_tolerance(None, tokens) is None


def test_snapshot_bytes_are_stat_measured_from_the_snapshot_dir(tmp_path):
    """A successful save stats the on-disk snapshot file (real bytes)."""
    snapshot_dir = tmp_path / "slots"
    snapshot_dir.mkdir()
    stub = StubClient()
    record = _run(
        m.measure_band(
            BAND_10K,
            client=stub,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="size-test",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            snapshot_dir=snapshot_dir,
        )
    )
    # The stub "wrote" no file, so bytes are recorded as null (not fabricated).
    assert record.snapshot_bytes is None
    # Now write the file the stub save would have created: the harness layout
    # names it deterministically, so the record's expected size is checkable.
    filename = "snapshot-bench-10k-size-test.bin"
    written = snapshot_dir / filename
    written.write_bytes(b"x" * m.expected_snapshot_bytes(record.prompt_total_tokens))
    stub2 = StubClient()
    record2 = _run(
        m.measure_band(
            BAND_10K,
            client=stub2,
            base_url="http://localhost:8080",
            slot_id=0,
            session_id="s",
            salt="size-test",
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            snapshot_dir=snapshot_dir,
        )
    )
    assert record2.snapshot_bytes == written.stat().st_size
    assert record2.snapshot_size_ratio == pytest.approx(1.0)
    assert record2.snapshot_size_in_tolerance is True


# ---------------------------------------------------------------------------
# Dry-run and CLI (AC: --bands / --dry-run / --json, no live server)
# ---------------------------------------------------------------------------


def test_dry_run_requires_no_http_and_no_live_server(capsys):
    exit_code = m.main(["--dry-run", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["records"] == []
    assert len(payload["bands"]) == 4
    for plan, band in zip(payload["bands"], m.DEFAULT_BANDS):
        assert plan["band"] == band.name
        assert plan["prompt_tokens_estimated"] >= band.target_tokens
    assert payload["timeout_model"]["per_token_seconds"] == pytest.approx(0.0015)


def test_dry_run_plan_timeouts_track_the_adaptive_model():
    plans = m.plan_for(
        m.DEFAULT_BANDS, salt="x", timeout_model=TIMEOUT_MODEL, under_load=True
    )
    expected = [
        TIMEOUT_MODEL.timeout_for(p.prompt_tokens_estimated) for p in plans
    ]
    assert [p.timeout_seconds for p in plans] == expected
    assert all(p.under_load for p in plans)
    # The 60k+ band is capped at 60 s in this model.
    assert plans[-1].timeout_seconds == pytest.approx(60.0)


def test_cli_wires_populate_and_slot_pinning_flags(capsys):
    """`--populate-slot` / `--verify-slot-id` / `--load-slot-id` reach the run."""
    exit_code = m.main(
        [
            "--dry-run",
            "--json",
            "--slot-id",
            "1",
            "--under-load",
            "--populate-slot",
            "--verify-slot-id",
            "1",
            "--load-slot-id",
            "0",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["populate_slot"] is True
    assert payload["slot_id"] == 1
    assert payload["verify_slot_id"] == 1
    assert payload["load_slot_id"] == 0
    assert payload["schema_version"] == 3


def test_cli_json_roundtrip_of_a_full_stub_run(capsys, tmp_path):
    """The CLI end-to-end (stub client) emits the full schema."""
    snapshot_dir = tmp_path / "slots"
    snapshot_dir.mkdir()

    class ScriptStubClient(StubClient):
        def __init__(self):
            super().__init__(
                verification={
                    "usage": {"prompt_tokens": 12_000},
                    "timings": {"prompt_n": 100},
                }
            )

    stub = ScriptStubClient()
    salt = "cli-test"
    result = _run(
        m.run_benchmark(
            bands=(BAND_10K,),
            base_url="http://localhost:8080",
            slot_id=3,
            session_id="cli",
            salt=salt,
            model=MODEL,
            timeout_model=TIMEOUT_MODEL,
            snapshot_dir=snapshot_dir,
            verify_chat=True,
            client=stub,
        )
    )
    payload = result.to_json()
    assert payload["totals"]["bands"] == 1
    assert payload["totals"]["save_success"] == 1
    record = payload["records"][0]
    assert set(record) == set(m.RECORD_FIELDS)
    assert record["cached_tokens"] == 12_000 - 100
    assert record["slot_id"] == 3
    # The JSON result roundtrips through the serializer the CLI writes.
    assert json.loads(json.dumps(payload)) == payload


def test_cli_rejects_unknown_bands(capsys):
    assert m.main(["--bands", "5k,zzz", "--dry-run"]) == 2
    assert "unknown band" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# AC5 guard: the harness is an analysis tool, never production code
# ---------------------------------------------------------------------------


def test_harness_is_not_imported_by_any_production_module():
    import importlib

    production_modules = (
        "proxy.session",
        "proxy.router",
        "proxy.handlers",
        "proxy.prefill_eta",
        "proxy.lifecycle",
        "proxy.metrics",
    )
    for name in production_modules:
        module = importlib.import_module(name)
        assert not hasattr(module, m.SNAPSHOT_BENCHMARK_SENTINEL), (
            f"{name} imported the snapshot benchmark harness — it is an "
            "analysis tool and must stay out of the production import graph"
        )
