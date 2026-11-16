#!/usr/bin/env python3
"""Tests for the weight-quantisation cost model (LP-0MWCEHYOO002TRGD).

The model decides a costing question the spike's AC1 table quotes, so these
tests pin the parts a later edit could quietly change:

- the GGUF tensor-table reader (synthetic files, not the 25 GiB models on this
  host) and its byte arithmetic against ggml's own block/byte table;
- llama.cpp's per-tensor quantisation rules as ported here: which tensors are
  quantisable at all, the category mapping, `use_more_bits`, and the preset
  type each real tensor name receives under Q4_K_S / Q4_K_M / Q5_K_M;
- the derived layout arithmetic (parameters, bytes, bits-per-weight, per-role
  shares) and the MTP nextn-block exclusion that makes the measured Q4_K_S
  comparable with the shipped non-MTP model;
- the throughput bounds: the weight-movement share of a prefill pass, the
  batch-1 active bytes per decode token, and the per-candidate gain/capacity
  arithmetic;
- the committed register: it re-verifies from its own embedded inputs, a
  tampered register is reported as drift, and the headline figures the document
  quotes still match the register.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quantisation_cost_model import (  # noqa: E402
    TYPE_BYTES,
    GgufError,
    active_bytes_per_decode_token,
    all_ok,
    allows_quantization,
    build_register,
    candidates,
    checks,
    is_mtp_tensor,
    layout_summary,
    main,
    predict_preset,
    prefill_weight_share,
    preset_type,
    read_gguf_tensor_table,
    tensor_bytes,
    tensor_category,
    use_more_bits,
)

REGISTER_PATH = REPO_ROOT / "docs" / "dev" / "quantisation-cost-2026-11-11.json"
DOC_PATH = REPO_ROOT / "docs" / "dev" / "quantisation-cost-2026-11-11.md"
GIB = 2**30


# ---------------------------------------------------------------------------
# Synthetic GGUF files
# ---------------------------------------------------------------------------
def _gguf_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, tensors: list[tuple[str, int, list[int]]]) -> Path:
    """Write a minimal but well-formed GGUF v3 file with the given tensors."""
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", 3)  # three metadata entries, one of each kind

    out += _gguf_string("general.architecture")
    out += struct.pack("<I", 8)  # STRING
    out += _gguf_string("qwen35moe")

    out += _gguf_string("qwen35moe.block_count")
    out += struct.pack("<I", 4)  # UINT32
    out += struct.pack("<I", 40)

    out += _gguf_string("qwen35moe.expert_used_count")
    out += struct.pack("<I", 9)  # ARRAY
    out += struct.pack("<I", 4)  # of UINT32
    out += struct.pack("<Q", 2)
    out += struct.pack("<I", 8) + struct.pack("<I", 256)

    offset = 0
    for name, ggml_type, dims in tensors:
        n = 1
        for d in dims:
            n *= d
        out += _gguf_string(name)
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", offset)
        block, size = TYPE_BYTES[
            {8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 0: "F32"}[ggml_type]
        ]
        offset += (n // block) * size
    path.write_bytes(bytes(out))
    return path


@pytest.fixture()
def small_gguf(tmp_path: Path) -> Path:
    # 256-weight tensors so every k-quant block divides exactly.
    return write_gguf(
        tmp_path / "small.gguf",
        [
            ("token_embd.weight", 8, [256, 2]),  # Q8_0, 512 weights
            ("blk.0.ffn_down_exps.weight", 12, [256, 4]),  # Q4_K, 1024 weights
            ("blk.0.attn_norm.weight", 0, [256]),  # F32
        ],
    )


# ---------------------------------------------------------------------------
# GGUF reader and byte arithmetic
# ---------------------------------------------------------------------------
class TestGgufReader:
    def test_reads_tensor_table(self, small_gguf: Path) -> None:
        tensors = read_gguf_tensor_table(small_gguf)
        assert [t["name"] for t in tensors] == [
            "token_embd.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_norm.weight",
        ]
        assert [t["type"] for t in tensors] == ["Q8_0", "Q4_K", "F32"]
        assert [t["params"] for t in tensors] == [512, 1024, 256]
        assert tensors[1]["dims"] == [256, 4]

    def test_rejects_non_gguf(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.gguf"
        bogus.write_bytes(b"NOPE" + b"\x00" * 32)
        with pytest.raises(GgufError, match="not a GGUF"):
            read_gguf_tensor_table(bogus)

    def test_rejects_unknown_type(self, tmp_path: Path) -> None:
        path = tmp_path / "x.gguf"
        path.write_bytes(
            b"GGUF"
            + struct.pack("<I", 3)
            + struct.pack("<Q", 1)
            + struct.pack("<Q", 0)
            + _gguf_string("t.weight")
            + struct.pack("<I", 1)
            + struct.pack("<Q", 256)
            + struct.pack("<I", 999)  # not a ggml type
            + struct.pack("<Q", 0)
        )
        with pytest.raises(GgufError, match="unknown ggml type 999"):
            read_gguf_tensor_table(path)

    @pytest.mark.parametrize(
        ("ggml_type", "params", "expected"),
        [
            ("Q4_K", 256, 144),
            ("Q5_K", 256, 176),
            ("Q6_K", 256, 210),
            ("Q8_0", 32, 34),
            ("F32", 7, 28),
        ],
    )
    def test_tensor_bytes(self, ggml_type: str, params: int, expected: int) -> None:
        assert tensor_bytes(ggml_type, params) == expected

    def test_tensor_bytes_rejects_ragged_block(self) -> None:
        with pytest.raises(GgufError, match="not divisible"):
            tensor_bytes("Q4_K", 200)


# ---------------------------------------------------------------------------
# llama.cpp's quantisation rules
# ---------------------------------------------------------------------------
class TestQuantisationRules:
    @pytest.mark.parametrize(
        ("name", "n_dims", "allowed"),
        [
            ("blk.3.ffn_down_exps.weight", 3, True),
            ("blk.3.attn_v.weight", 2, True),
            ("output.weight", 2, True),
            ("blk.3.attn_norm.weight", 1, False),
            ("blk.3.ffn_norm.weight", 1, False),
            ("blk.3.ffn_gate_inp.weight", 2, False),
            ("blk.3.ssm_conv1d.weight", 3, False),
            ("blk.3.ssm_a", 1, False),
            ("blk.40.nextn.eh_proj.weight", 2, True),
        ],
    )
    def test_allows_quantization(self, name: str, n_dims: int, allowed: bool) -> None:
        assert allows_quantization(name, n_dims) is allowed

    @pytest.mark.parametrize(
        ("name", "category"),
        [
            ("output.weight", "OUTPUT"),
            ("token_embd.weight", "TOKEN_EMBD"),
            ("blk.0.attn_qkv.weight", "ATTN_QKV"),
            ("blk.30.attn_v.weight", "ATTN_V"),
            ("blk.30.attn_q.weight", "ATTN_Q"),
            ("blk.30.attn_k.weight", "ATTN_K"),
            ("blk.30.attn_output.weight", "ATTN_OUTPUT"),
            ("blk.5.ffn_down_exps.weight", "FFN_DOWN"),
            ("blk.5.ffn_down_shexp.weight", "FFN_DOWN"),
            ("blk.5.ffn_gate_exps.weight", "FFN_GATE"),
            ("blk.5.ffn_up_exps.weight", "FFN_UP"),
            ("blk.5.ffn_gate_inp.weight", "FFN_GATE"),
            ("blk.5.ssm_out.weight", "OTHER"),
        ],
    )
    def test_tensor_category(self, name: str, category: str) -> None:
        assert tensor_category(name) == category

    def test_use_more_bits_matches_llama_cpp_for_this_model(self) -> None:
        # n_layers=40 -> n/8 = 5: first 5, last 5, and (i-5)%3==2.
        assert [i for i in range(40) if use_more_bits(i, 40)] == [
            0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39
        ]
        # n_layers=80 (40 blocks x down_exps + down_shexp): n/8 = 10.
        assert [i for i in range(40) if use_more_bits(i, 80)] == [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39
        ]

    @pytest.mark.parametrize(
        ("name", "dims", "preset", "expected"),
        [
            # attn-v-like tensors: _S upgrades only the first four layers.
            ("blk.0.attn_qkv.weight", [256, 2], "q4_k_s", "Q5_K"),
            ("blk.3.attn_v.weight", [256, 2], "q4_k_s", "Q5_K"),
            ("blk.4.attn_v.weight", [256, 2], "q4_k_s", "Q4_K"),
            ("blk.0.attn_qkv.weight", [256, 2], "q4_k_m", "Q6_K"),
            ("blk.5.attn_v.weight", [256, 2], "q4_k_m", "Q4_K"),
            # ffn_down: _S upgrades the first tenth; _M uses use_more_bits.
            ("blk.9.ffn_down_exps.weight", [256, 2], "q4_k_s", "Q5_K"),
            ("blk.10.ffn_down_exps.weight", [256, 2], "q4_k_s", "Q4_K"),
            ("blk.12.ffn_down_shexp.weight", [256, 2], "q4_k_m", "Q6_K"),
            ("blk.11.ffn_down_shexp.weight", [256, 2], "q4_k_m", "Q4_K"),
            # gate/up keep the preset's default type.
            ("blk.3.ffn_gate_exps.weight", [256, 2], "q4_k_m", "Q4_K"),
            ("blk.3.ffn_up_exps.weight", [256, 2], "q5_k_m", "Q5_K"),
            # output is bumped to Q6_K, embeddings keep the default.
            ("output.weight", [256, 2], "q4_k_s", "Q6_K"),
            ("output.weight", [256, 2], "q5_k_m", "Q6_K"),
            ("token_embd.weight", [256, 2], "q4_k_m", "Q4_K"),
            ("token_embd.weight", [256, 2], "q5_k_m", "Q5_K"),
        ],
    )
    def test_preset_type(self, name, dims, preset, expected) -> None:
        assert preset_type(name, dims, preset) == expected

    def test_preset_type_leaves_unquantisable_tensors_alone(self) -> None:
        assert preset_type("blk.0.attn_norm.weight", [256], "q4_k_m") is None
        assert preset_type("blk.0.ffn_gate_inp.weight", [256, 256], "q4_k_m") is None

    def test_preset_type_rejects_unknown_preset(self) -> None:
        with pytest.raises(ValueError, match="unknown preset"):
            preset_type("blk.0.attn_v.weight", [256, 2], "q3_k_xl")


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def _tensor(name: str, ggml_type: str, params: int) -> dict:
    return {"name": name, "type": ggml_type, "params": params, "dims": [256, 2]}


class TestLayoutSummary:
    def test_totals_and_roles(self) -> None:
        tensors = [
            _tensor("blk.0.ffn_gate_exps.weight", "Q5_K", 256),
            _tensor("blk.0.ffn_up_exps.weight", "Q5_K", 256),
            _tensor("blk.0.ffn_down_exps.weight", "Q6_K", 256),
            _tensor("token_embd.weight", "Q8_0", 256),
            _tensor("blk.0.attn_norm.weight", "F32", 256),
        ]
        summary = layout_summary(tensors)
        assert summary["params"] == 1280
        assert summary["weight_bytes"] == 176 + 176 + 210 + 272 + 1024
        assert summary["bits_per_weight"] == pytest.approx(
            summary["weight_bytes"] * 8 / 1280, abs=1e-4
        )
        assert set(summary["role_bytes"]) == {
            "expert_gate",
            "expert_up",
            "expert_down",
            "token_embd",
            "norm",
        }
        assert summary["role_bytes"]["norm"]["share_pct"] == pytest.approx(
            100 * 1024 / summary["weight_bytes"], abs=1e-3
        )

    def test_mtp_block_is_detectable_and_excludable(self) -> None:
        assert is_mtp_tensor("blk.40.attn_q.weight")
        assert not is_mtp_tensor("blk.39.attn_q.weight")
        tensors = [
            _tensor("blk.0.attn_v.weight", "Q8_0", 256),
            _tensor("blk.40.attn_v.weight", "Q8_0", 256),
        ]
        assert layout_summary(tensors)["params"] == 512
        assert layout_summary(tensors, exclude_mtp=True)["params"] == 256


class TestPredictPreset:
    def test_presets_order_by_size(self) -> None:
        tensors = [
            _tensor(f"blk.{i}.ffn_down_exps.weight", "Q5_K", 256) for i in range(40)
        ] + [
            _tensor(f"blk.{i}.ffn_gate_exps.weight", "Q5_K", 256) for i in range(40)
        ] + [_tensor("output.weight", "Q8_0", 256)]
        sizes = {p: predict_preset(tensors, p)["weight_bytes"] for p in
                 ("q5_k_m", "q4_k_m", "q4_k_s")}
        assert sizes["q4_k_s"] < sizes["q4_k_m"] < sizes["q5_k_m"]

    def test_unquantisable_tensors_keep_their_measured_type(self) -> None:
        tensors = [_tensor("blk.0.attn_norm.weight", "F32", 256)]
        assert predict_preset(tensors, "q4_k_s")["weight_bytes"] == 1024

    def test_rejects_unknown_preset(self) -> None:
        with pytest.raises(ValueError):
            predict_preset([], "iq2_xxs")


# ---------------------------------------------------------------------------
# Throughput bounds
# ---------------------------------------------------------------------------
class TestPrefillWeightShare:
    def test_arithmetic(self) -> None:
        # 2048-token prompt -> 2 micro-batches; 1024 tokens/s -> 1.024 s each.
        info = prefill_weight_share((2048, 1024), 24 * GIB, 100e9)
        assert info["micro_batches"] == 2
        assert info["pass_seconds"] == pytest.approx(1.0, abs=1e-6)
        assert info["request_prefill_seconds"] == pytest.approx(2.0, abs=1e-6)
        # weight bytes / bandwidth = 24 GiB / 100 GB/s per pass.
        assert info["weight_ms_per_pass"] == pytest.approx(
            1000 * 24 * GIB / 100e9, rel=1e-6
        )
        assert info["weight_share_pct"] == pytest.approx(
            100 * (2 * 24 * GIB / 100e9) / 2.0, abs=1e-2
        )
        assert info["weight_stream_bytes_per_second"] == pytest.approx(
            24 * GIB / 1.0, rel=1e-2
        )

    def test_ceils_the_pass_count(self) -> None:
        assert prefill_weight_share((1025, 1024), GIB, 1e9)["micro_batches"] == 2
        assert prefill_weight_share((1024, 1024), GIB, 1e9)["micro_batches"] == 1

    def test_rejects_nonpositive_rate(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            prefill_weight_share((1000, 0), GIB, 1e9)


class TestDecodeTopology:
    def test_routes_eight_of_256_experts_and_excludes_embeddings(self) -> None:
        tensors = [
            *[_tensor("blk.0.ffn_gate_exps.weight", "Q4_K", 256 * 8) for _ in range(1)],
            _tensor("blk.0.ffn_down_shexp.weight", "Q4_K", 256),
            _tensor("blk.0.attn_qkv.weight", "Q8_0", 256),
            _tensor("token_embd.weight", "Q8_0", 256),
        ]
        layout = layout_summary(tensors)
        info = active_bytes_per_decode_token(tensors, layout)
        expert_bytes = tensor_bytes("Q4_K", 2048)
        assert info["routed_expert_bytes"] == int(expert_bytes / 256 * 8)
        assert info["shared_expert_bytes"] == tensor_bytes("Q4_K", 256)
        assert info["other_bytes"] == tensor_bytes("Q8_0", 256)
        assert info["active_bytes"] == (
            info["routed_expert_bytes"]
            + info["shared_expert_bytes"]
            + info["other_bytes"]
        )


class TestCandidates:
    def _buckets(self) -> list[dict]:
        return [
            {
                "bucket": "20k-40k",
                "mean_prompt_tokens": 20000.0,
                "measured_tokens_per_second": 240.0,
                "weight_share_pct": 7.0,
            }
        ]

    def test_gain_and_capacity_arithmetic(self) -> None:
        shipped = 100 * GIB
        candidate = 80 * GIB
        capacity = {
            "local_busy_seconds": 1200.0,
            "local_completion_tokens": 5000,
            "local_token_share_pct": 0.70,
            "local_stream_share_pct": 2.57,
            "decode_slot_share_pct": 13.4,
        }
        decode = {"tokens_per_second": 30.0, "slot_seconds": 5000 / 30.0}
        out = candidates(
            shipped, {"x": candidate}, self._buckets(), 100e9, decode, capacity
        )["x"]
        assert out["byte_ratio"] == 0.8
        assert out["delta_vs_shipped_gib"] == -20
        assert out["prefill"]["20k-40k"]["prefill_gain_ceiling_pct"] == 25.0
        assert out["prefill"]["20k-40k"]["prefill_gain_measured_share_pct"] == pytest.approx(
            1.75, abs=1e-3
        )
        assert out["decode"]["byte_ratio_bound_pct"] == 25.0
        assert out["decode"]["slot_seconds_freed"] == pytest.approx(
            5000 / 30.0 - (5000 / 30.0) / 1.25, abs=0.05
        )
        exact_freed = 5000 / 30.0 - (5000 / 30.0) / 1.25
        expected_capacity_gain = 100 * exact_freed / 1200.0
        assert out["capacity"]["free_slot_seconds_gain_pct"] == pytest.approx(
            expected_capacity_gain, abs=1e-3
        )
        assert out["capacity"]["local_token_share_after_pct"] == pytest.approx(
            0.70 * (1 + expected_capacity_gain / 100), abs=1e-4
        )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def register() -> dict:
    return json.loads(REGISTER_PATH.read_text())


class TestCommittedRegister:
    def test_register_verifies(self, register: dict) -> None:
        results = checks(register)
        assert all_ok(results), [r for r in results if not r["ok"]]
        assert len(results) >= 30

    def test_register_identity(self, register: dict) -> None:
        assert register["work_item"] == "LP-0MWCEHYOO002TRGD"
        assert register["parent_work_item"] == "LP-0MVXY8ZBZ0003042"
        assert register["read_only"] is True
        assert register["engine"]["shipped_preset"] == "UD-Q5_K_M"
        assert register["engine"]["ubatch_tokens"] == 1024

    def test_tampered_register_is_reported_as_drift(self, register: dict) -> None:
        tampered = json.loads(json.dumps(register))
        tampered["candidates"]["ud_q4_k_s_measured"]["weight_bytes"] *= 1.02
        results = checks(tampered)
        assert not all_ok(results)
        drifted = [r for r in results if not r["ok"]]
        assert any("byte ratio" in r["claim"] for r in drifted)

    def test_candidates_are_bounded_below_by_zero_and_above_by_the_ratio(
        self, register: dict
    ) -> None:
        for name, cand in register["candidates"].items():
            assert cand["byte_ratio"] < 1.0
            for bucket in cand["prefill"].values():
                assert bucket["prefill_gain_measured_share_pct"] >= 0.0
                assert (
                    bucket["prefill_gain_measured_share_pct"]
                    <= bucket["prefill_gain_ceiling_pct"]
                )
            assert cand["decode"]["byte_ratio_bound_pct"] > 0.0

    def test_weight_streaming_is_far_below_the_memory_subsystem(
        self, register: dict
    ) -> None:
        """The refutation the report rests on: prefill streams only ~6 GB/s."""
        for bucket in register["inputs"]["prefill"]["buckets"]:
            assert bucket["weight_stream_bytes_per_second"] < 10e9
            assert bucket["weight_share_pct"] < 10.0

    def test_cli_check_exits_zero(self, capsys) -> None:
        assert main(["--register", str(REGISTER_PATH), "--check", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)

    def test_cli_check_fails_on_drift(self, register: dict, tmp_path: Path) -> None:
        tampered = json.loads(json.dumps(register))
        tampered["inputs"]["capacity"]["decode_slot_share_pct"] = 99.0
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(tampered))
        assert main(["--register", str(path), "--check"]) == 1


class TestDocumentFidelity:
    """The document is the deliverable: its quoted figures must be the
    register's, so a later edit to either cannot leave the pair inconsistent."""

    def test_headline_figures_appear_in_the_document(self, register: dict) -> None:
        doc = DOC_PATH.read_text()
        shipped_gib = register["layouts"]["shipped_ud_q5_k_m"]["gib"]
        q4m = register["candidates"]["ud_q4_k_m_estimated"]
        q4s = register["candidates"]["ud_q4_k_s_measured"]
        capacity = register["inputs"]["capacity"]

        assert f"{shipped_gib:.3f} GiB" in doc
        assert f"{q4m['gib']:.3f} GiB" in doc
        assert f"{q4s['gib']:.3f} GiB" in doc
        assert f"{abs(q4m['delta_vs_shipped_gib']):.3f} GiB" in doc
        assert f"{abs(q4s['delta_vs_shipped_gib']):.3f} GiB" in doc
        assert f"{capacity['decode_slot_share_pct']:.1f} %" in doc
        assert f"{capacity['local_token_share_pct']:.3f} %" in doc

    def test_every_candidate_and_bucket_name_is_documented(self, register: dict) -> None:
        doc = DOC_PATH.read_text()
        assert "UD-Q4_K_M" in doc and "UD-Q4_K_S" in doc
        labels = {
            "lt8k": "< 8 k",
            "8k-20k": "8 k–20 k",
            "20k-40k": "20 k–40 k",
            "40k-80k": "40 k–80 k",
        }
        for bucket in register["inputs"]["prefill"]["buckets"]:
            assert labels[bucket["bucket"]] in doc

    def test_owners_of_the_other_levers_are_cited(self) -> None:
        doc = DOC_PATH.read_text()
        for owner in (
            "LP-0MVNBJXR6002OSPD",
            "LP-0MVNBJY2R004EZ51",
            "LP-0MVNCMEJ6004A2QY",
            "LP-0MVMHNUXW0061FGN",
            "LP-0MSAOQTJS000FFVM",
        ):
            assert owner in doc

    def test_document_is_read_only_and_states_the_quality_caveat(self) -> None:
        doc = DOC_PATH.read_text()
        assert "read-only" in doc.lower()
        assert "quality" in doc.lower()
