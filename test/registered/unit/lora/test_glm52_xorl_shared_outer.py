"""CPU contract tests for the complete XoRL -> GLM-5.2 LoRA path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import argparse
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.lora import lora_moe_runners
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.glm52 import (
    GLM52_LOGICAL_FACTOR_COUNT,
    GLM52_REQUIRED_TARGET_MODULES,
    Glm52XorlSharedOuterValidator,
    build_glm52_xorl_shared_outer_inventory,
    maybe_create_glm52_validator,
    summarize_glm52_factor_roles,
)
from sglang.srt.lora.layers import (
    FusedMoEWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import LoRABatchInfo, get_normalized_target_modules
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM
from sglang.srt.runtime_context import get_forward
from sglang.srt.server_args import ServerArgs


def _make_cp_lora_manager(batch_size, lengths=None):
    lengths = lengths or [4096] * batch_size
    global_seg_lens = torch.tensor(lengths, dtype=torch.int32)
    global_seg_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(global_seg_lens, dim=0)]
    )
    weight_indices = torch.arange(batch_size, dtype=torch.int32)
    full_moe_info = object()
    full_batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=batch_size,
        num_segments=batch_size,
        seg_indptr=global_seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=torch.full((2,), 16, dtype=torch.int64),
        scalings=torch.ones(2),
        max_len=max(lengths),
        seg_lens=global_seg_lens,
        permutation=None,
        expected_tokens=None,
        has_active_lora=True,
        moe_lora_info=full_moe_info,
    )
    full_lm_head_info = SimpleNamespace(expected_tokens=1)
    full_lm_head_pass_infos = [SimpleNamespace(expected_tokens=1)]

    def add_moe_lora_info(_forward_batch, batch_info):
        batch_info.moe_lora_info = SimpleNamespace(
            seg_indptr=batch_info.req_seg_indptr,
            req_to_lora=batch_info.req_weight_indices,
            token_lora_mapping=torch.repeat_interleave(
                batch_info.req_weight_indices, batch_info.seg_lens
            ),
        )
        return batch_info

    backend = SimpleNamespace(
        name="triton",
        is_moe_lora=True,
        batch_info=full_batch_info,
        context_parallel_mlp_batch_info=None,
        sgemm_batch_info=None,
        lm_head_batch_info=full_lm_head_info,
        lm_head_pass_batch_infos=full_lm_head_pass_infos,
        _lm_head_pass_idx=0,
        _add_moe_lora_info=add_moe_lora_info,
    )

    def get_batch_info_for_rows(num_tokens):
        if backend.batch_info.expected_tokens == num_tokens:
            return backend.batch_info
        if (
            backend.context_parallel_mlp_batch_info is not None
            and backend.context_parallel_mlp_batch_info.expected_tokens == num_tokens
        ):
            return backend.context_parallel_mlp_batch_info
        raise RuntimeError(f"no metadata for {num_tokens} rows")

    backend.get_batch_info_for_rows = get_batch_info_for_rows
    manager = object.__new__(LoRAManager)
    manager.lora_backend = backend
    manager.base_hf_config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    manager.tp_size = 16
    manager.dp_size = 1
    manager.ep_size = 16
    manager.pp_size = 1
    manager.attn_cp_size = 16
    manager.cp_strategy = "interleave"
    manager._experts_shared_outer_override = True
    manager.enable_dp_attention = True
    return (
        manager,
        backend,
        full_batch_info,
        full_moe_info,
        full_lm_head_info,
        full_lm_head_pass_infos,
    )


def _cp_forward_batch(batch_size, lengths=None):
    lengths = lengths or [4096] * batch_size
    max_logical_rank_tokens = (sum(lengths) + 15) // 16
    physical_rank_tokens = (max_logical_rank_tokens + 15) // 16 * 16
    return SimpleNamespace(
        batch_size=batch_size,
        lora_ids=[f"adapter-{index}" for index in range(batch_size)],
        extend_num_tokens=sum(lengths),
        extend_seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
        return_logprob=True,
        extend_logprob_start_lens_cpu=[-1] * batch_size,
        attn_cp_metadata=SimpleNamespace(
            per_rank_actual_token=[physical_rank_tokens] * 16
        ),
    )


def test_cp_lora_batch_installs_ab_segments_and_restores_full_state():
    (
        manager,
        backend,
        full_batch_info,
        full_moe_info,
        full_lm_head_info,
        full_lm_head_pass_infos,
    ) = _make_cp_lora_manager(batch_size=2)
    forward_batch = _cp_forward_batch(batch_size=2)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)

    with pytest.raises(RuntimeError, match="body failure"):
        with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
            with manager.glm52_context_parallel_lora_batch(
                forward_batch, local_num_tokens=512
            ):
                local_batch_info = backend.batch_info
                assert local_batch_info is not full_batch_info
                assert local_batch_info.expected_tokens == 512
                assert local_batch_info.seg_lens.tolist() == [256, 256]
                assert local_batch_info.seg_indptr.tolist() == [0, 256, 512]
                assert local_batch_info.req_weight_indices.tolist() == [0, 1]
                assert local_batch_info.moe_lora_info.token_lora_mapping.tolist() == (
                    [0] * 256 + [1] * 256
                )
                gathered_batch_info = backend.context_parallel_mlp_batch_info
                assert gathered_batch_info.expected_tokens == 8192
                assert gathered_batch_info.seg_lens.tolist() == [256, 256] * 16
                assert backend.get_batch_info_for_rows(512) is local_batch_info
                assert backend.get_batch_info_for_rows(8192) is gathered_batch_info
                assert backend.lm_head_batch_info is full_lm_head_info
                assert backend.lm_head_pass_batch_infos is full_lm_head_pass_infos
                raise RuntimeError("body failure")

    assert backend.batch_info is full_batch_info
    assert backend.batch_info.moe_lora_info is full_moe_info
    assert backend.context_parallel_mlp_batch_info is None
    assert backend.lm_head_batch_info is full_lm_head_info
    assert backend.lm_head_pass_batch_infos is full_lm_head_pass_infos
    assert backend._lm_head_pass_idx == 0


def test_cp_lora_batch_installs_single_request_local_segment():
    manager, backend, full_batch_info, *_ = _make_cp_lora_manager(batch_size=1)
    forward_batch = _cp_forward_batch(batch_size=1)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)

    with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
        with manager.glm52_context_parallel_lora_batch(
            forward_batch, local_num_tokens=256
        ):
            assert backend.batch_info.seg_lens.tolist() == [256]
            assert backend.batch_info.seg_indptr.tolist() == [0, 256]
            assert (
                backend.batch_info.moe_lora_info.token_lora_mapping.tolist()
                == [0] * 256
            )
            assert backend.context_parallel_mlp_batch_info.expected_tokens == 4096
            assert (
                backend.context_parallel_mlp_batch_info.seg_lens.tolist() == [256] * 16
            )
    assert backend.batch_info is full_batch_info


@pytest.mark.parametrize(
    ("cp_rank", "expected_lens", "expected_weights"),
    [
        (0, [256, 16], [0] * 256 + [1] * 16),
        (15, [272], [0] * 272),
    ],
)
def test_cp_lora_batch_handles_mixed_cold_warm_requests_and_padding(
    cp_rank, expected_lens, expected_weights
):
    lengths = [4096, 1]
    manager, backend, full_batch_info, *_ = _make_cp_lora_manager(
        batch_size=2, lengths=lengths
    )
    forward_batch = _cp_forward_batch(batch_size=2, lengths=lengths)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=cp_rank)
    with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
        with manager.glm52_context_parallel_lora_batch(
            forward_batch, local_num_tokens=272
        ):
            assert backend.batch_info.bs == len(expected_lens)
            assert backend.batch_info.num_segments == len(expected_lens)
            assert backend.batch_info.expected_tokens == 272
            assert backend.batch_info.seg_lens.tolist() == expected_lens
            assert backend.batch_info.seg_indptr.tolist() == [
                0,
                *torch.cumsum(torch.tensor(expected_lens), dim=0).tolist(),
            ]
            assert backend.batch_info.moe_lora_info.token_lora_mapping.tolist() == (
                expected_weights
            )
            gathered_batch_info = backend.context_parallel_mlp_batch_info
            assert gathered_batch_info.expected_tokens == 4352
            assert gathered_batch_info.seg_lens.tolist() == [256, 16] + [272] * 15
            assert gathered_batch_info.moe_lora_info.token_lora_mapping.tolist() == [
                0
            ] * 256 + [1] * 16 + [0] * (272 * 15)
    assert backend.batch_info is full_batch_info


def test_cp_lora_batch_rejects_nonpositive_extend_lengths():
    lengths = [4096, 0]
    manager, _, *_ = _make_cp_lora_manager(batch_size=2, lengths=lengths)
    forward_batch = _cp_forward_batch(batch_size=2, lengths=lengths)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)
    with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
        with pytest.raises(
            RuntimeError, match="one positive extend length per request"
        ):
            with manager.glm52_context_parallel_lora_batch(
                forward_batch, local_num_tokens=256
            ):
                pass


def test_cp_lora_batch_rejects_physical_metadata_mismatch():
    manager, _, *_ = _make_cp_lora_manager(batch_size=1)
    forward_batch = _cp_forward_batch(batch_size=1)
    forward_batch.attn_cp_metadata.per_rank_actual_token[-1] = 272
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)
    with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
        with pytest.raises(RuntimeError, match="physical CP metadata"):
            with manager.glm52_context_parallel_lora_batch(
                forward_batch, local_num_tokens=256
            ):
                pass


def test_cp_lora_batch_rejects_uncertified_world_geometry():
    manager, _, *_ = _make_cp_lora_manager(batch_size=1)
    manager.tp_size = 8
    forward_batch = _cp_forward_batch(batch_size=1)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)
    with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
        with pytest.raises(RuntimeError, match="only certified for the WORLD16"):
            with manager.glm52_context_parallel_lora_batch(
                forward_batch, local_num_tokens=256
            ):
                pass


@pytest.mark.parametrize(
    ("patch_target", "patch_value"),
    [
        (
            "sglang.srt.lora.lora_manager._SGLANG_EXPERIMENTAL_LORA_OPTI",
            True,
        ),
        (
            "sglang.srt.layers.moe.get_moe_a2a_backend",
            lambda: SimpleNamespace(value="deepep"),
        ),
    ],
)
def test_cp_lora_batch_rejects_row_reordering_modes(patch_target, patch_value):
    manager, _, *_ = _make_cp_lora_manager(batch_size=1)
    forward_batch = _cp_forward_batch(batch_size=1)
    strategy = SimpleNamespace(name="interleave", cp_size=16, cp_rank=0)
    with (
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
        patch(patch_target, patch_value),
        pytest.raises(RuntimeError, match="only certified for the WORLD16"),
    ):
        with manager.glm52_context_parallel_lora_batch(
            forward_batch, local_num_tokens=256
        ):
            pass


def test_triton_lora_rejects_global_segments_for_cp_local_rows():
    batch_info = SimpleNamespace(expected_tokens=4096)
    with pytest.raises(RuntimeError, match="metadata_rows=4096, activation_rows=256"):
        TritonLoRABackend._validate_sgemm_rows(torch.empty(256, 4), batch_info)


def test_triton_lora_selects_cp_gathered_metadata_only_for_gathered_rows():
    local_batch_info = SimpleNamespace(expected_tokens=256)
    gathered_batch_info = SimpleNamespace(expected_tokens=4096)
    backend = object.__new__(TritonLoRABackend)
    backend.batch_info = local_batch_info
    backend.context_parallel_mlp_batch_info = gathered_batch_info
    backend.sgemm_batch_info = None

    assert backend._sgemm_info(num_tokens=256) is local_batch_info
    assert backend._sgemm_info(num_tokens=4096) is gathered_batch_info
    assert backend._sgemm_info(num_tokens=256) is local_batch_info
    with pytest.raises(RuntimeError, match="activation_rows=1024"):
        backend._sgemm_info(num_tokens=1024)
    with get_forward().scoped(mlp_reduce_scatter=True):
        assert backend._sgemm_info(num_tokens=256) is local_batch_info
        assert backend._sgemm_info(num_tokens=4096) is gathered_batch_info


def test_fused_moe_uses_cp_gathered_metadata_and_row_guard():
    local_batch_info = SimpleNamespace(expected_tokens=256)
    gathered_batch_info = SimpleNamespace(expected_tokens=4096)
    backend = object.__new__(TritonLoRABackend)
    backend.batch_info = local_batch_info
    backend.context_parallel_mlp_batch_info = gathered_batch_info
    backend.sgemm_batch_info = None
    seen = []

    wrapper = SimpleNamespace(
        lora_backend=backend,
        base_layer=SimpleNamespace(forward=lambda *args, **kwargs: None),
        _get_lora_info=lambda batch_info: seen.append(batch_info) or object(),
        _forward_with_lora=lambda hidden_states, topk_output, lora_info, **kwargs: (
            hidden_states
        ),
    )
    hidden_states = torch.empty(4096, 4)

    with pytest.raises(RuntimeError, match="activation_rows=1024"):
        FusedMoEWithLoRA.forward(wrapper, torch.empty(1024, 4), object())
    assert FusedMoEWithLoRA.forward(wrapper, hidden_states, object()) is hidden_states
    assert seen == [gathered_batch_info]


def test_decode_graph_refreshes_fixed_sgemm_routing_outside_cp_extend():
    assert ForwardMode.DECODE.is_cuda_graph()
    assert not ForwardMode.DECODE.is_context_parallel_extend()

    backend = object.__new__(TritonLoRABackend)
    backend.max_loras_per_batch = 3
    backend.batch_info = LoRABatchInfo(
        use_cuda_graph=True,
        bs=4,
        num_segments=4,
        seg_indptr=torch.arange(5, dtype=torch.int32),
        weight_indices=torch.tensor([2, 1, 2, 0], dtype=torch.int32),
        lora_ranks=torch.tensor([0, 2, 2], dtype=torch.int32),
        scalings=torch.tensor([0.0, 0.5, 1.25]),
        max_len=1,
        seg_lens=torch.ones(4, dtype=torch.int32),
        permutation=None,
    )
    fixed_sgemm_info = LoRABatchInfo(
        use_cuda_graph=True,
        bs=3,
        num_segments=3,
        seg_indptr=torch.zeros(4, dtype=torch.int32),
        weight_indices=torch.arange(3, dtype=torch.int32),
        lora_ranks=torch.zeros(3, dtype=torch.int32),
        scalings=torch.zeros(3),
        max_len=4,
        seg_lens=torch.zeros(3, dtype=torch.int32),
        permutation=torch.zeros(4, dtype=torch.int32),
    )
    backend.cuda_graph_sgemm_batch_info = fixed_sgemm_info

    fixed_ptrs = (
        fixed_sgemm_info.seg_lens.data_ptr(),
        fixed_sgemm_info.seg_indptr.data_ptr(),
        fixed_sgemm_info.permutation.data_ptr(),
    )
    backend.compute_sgemm_routing(use_cuda_graph=True)
    assert backend.sgemm_batch_info is fixed_sgemm_info
    assert fixed_sgemm_info.permutation.tolist() == [3, 1, 0, 2]
    assert fixed_sgemm_info.seg_lens.tolist() == [1, 1, 2]
    assert fixed_sgemm_info.seg_indptr.tolist() == [0, 1, 2, 4]

    # Replay a smaller live batch. The graph-owned tensors and their addresses
    # stay fixed; only the live prefix and segment lengths change.
    backend.batch_info.bs = 2
    backend.batch_info.num_segments = 2
    backend.batch_info.weight_indices[:2].copy_(torch.tensor([1, 0]))
    backend.compute_sgemm_routing(use_cuda_graph=True)
    assert fixed_ptrs == (
        fixed_sgemm_info.seg_lens.data_ptr(),
        fixed_sgemm_info.seg_indptr.data_ptr(),
        fixed_sgemm_info.permutation.data_ptr(),
    )
    assert fixed_sgemm_info.permutation[:2].tolist() == [1, 0]
    assert fixed_sgemm_info.seg_lens.tolist() == [1, 1, 0]
    assert fixed_sgemm_info.seg_indptr.tolist() == [0, 1, 2, 2]


def test_decode_graph_refreshes_fixed_moe_routing_outside_cp_extend():
    backend = object.__new__(TritonLoRABackend)
    backend.max_loras_per_batch = 3
    backend._is_moe_lora = True
    adapter_enabled = torch.zeros(3, dtype=torch.int32)
    token_lora_mapping = torch.full((4,), -1, dtype=torch.int32)
    backend.moe_cg_buffers = {
        "adapter_enabled": adapter_enabled,
        "token_lora_mapping": token_lora_mapping,
    }
    batch_info = LoRABatchInfo(
        use_cuda_graph=True,
        bs=4,
        num_segments=4,
        seg_indptr=torch.arange(5, dtype=torch.int32),
        weight_indices=torch.tensor([2, 1, 2, 0], dtype=torch.int32),
        lora_ranks=torch.tensor([0, 2, 2], dtype=torch.int32),
        scalings=torch.tensor([0.0, 0.5, 1.25]),
        max_len=1,
        seg_lens=torch.ones(4, dtype=torch.int32),
        permutation=None,
    )
    adapter_ptr = adapter_enabled.data_ptr()
    mapping_ptr = token_lora_mapping.data_ptr()

    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
    )
    backend._add_moe_lora_info(forward_batch, batch_info)
    assert batch_info.moe_lora_info.adapter_enabled.tolist() == [0, 1, 1]
    assert batch_info.moe_lora_info.token_lora_mapping.tolist() == [2, 1, 2, 0]

    batch_info.bs = 2
    batch_info.num_segments = 2
    batch_info.weight_indices[:2].copy_(torch.tensor([1, 0]))
    forward_batch.batch_size = 2
    backend._add_moe_lora_info(forward_batch, batch_info)
    assert adapter_enabled.data_ptr() == adapter_ptr
    assert token_lora_mapping.data_ptr() == mapping_ptr
    assert batch_info.moe_lora_info.adapter_enabled.tolist() == [0, 1, 0]
    assert batch_info.moe_lora_info.token_lora_mapping.tolist() == [1, 0]


def test_decode_graph_capture_records_moe_lora_nodes_for_empty_synthetic_batch(
    monkeypatch,
):
    lora_info = lora_moe_runners.LoRAInfo(
        gate_up_lora_a_weights=torch.empty(1, 1, 2, 4),
        gate_up_lora_b_weights=torch.empty(1, 2, 6, 2),
        down_lora_a_weights=torch.empty(1, 2, 2, 3),
        down_lora_b_weights=torch.empty(1, 1, 4, 2),
        seg_indptr=torch.tensor([0, 2], dtype=torch.int32),
        req_to_lora=torch.tensor([0], dtype=torch.int32),
        lora_ranks=torch.tensor([0], dtype=torch.int32),
        adapter_enabled=torch.tensor([0], dtype=torch.int32),
        token_lora_mapping=torch.tensor([0, 0], dtype=torch.int32),
        max_lora_rank=2,
        num_experts=2,
        has_active_lora=False,
        lora_use_virtual_experts=True,
    )
    hidden_states = torch.empty(2, 4)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    monkeypatch.setattr(lora_moe_runners, "get_is_capture_mode", lambda: False)
    hooks = lora_moe_runners.build_lora_hooks(hidden_states, lora_info, topk_ids)
    assert hooks.after_gate_up is None
    assert hooks.after_down is None

    # Decode capture uses [None] * bucket as its synthetic LoRA IDs. Capture
    # mode must still install both callbacks so their kernels become graph
    # nodes; replay-time adapter_enabled values decide whether they do work.
    monkeypatch.setattr(lora_moe_runners, "get_is_capture_mode", lambda: True)
    hooks = lora_moe_runners.build_lora_hooks(hidden_states, lora_info, topk_ids)
    assert hooks.after_gate_up is not None
    assert hooks.after_down is not None


def _official_config():
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        first_k_dense_replace=3,
        hidden_size=6144,
        intermediate_size=12288,
        kv_lora_rank=512,
        moe_intermediate_size=2048,
        moe_layer_freq=1,
        n_routed_experts=256,
        n_shared_experts=1,
        num_attention_heads=64,
        num_hidden_layers=78,
        q_lora_rank=2048,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        vocab_size=154880,
    )


def _adapter_config():
    return {
        "_sglang_lora_format": "shared_outer",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 2,
        "moe_hybrid_shared_lora": True,
        "peft_type": "LORA",
        "r": 2,
        "target_modules": sorted(GLM52_REQUIRED_TARGET_MODULES),
        "task_type": "CAUSAL_LM",
    }


def _exact_adapter_config():
    adapter_config = _adapter_config()
    adapter_config.update(
        r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        use_dora=False,
        use_rslora=False,
        alpha_pattern={},
        rank_pattern={},
    )
    return adapter_config


def _exact_official_config():
    config = _official_config()
    config._glm52_exact_mode = True
    return config


def test_exact_inventory_accepts_only_the_rank_one_alpha_one_factor_program():
    specs = build_glm52_xorl_shared_outer_inventory(
        _exact_official_config(), _exact_adapter_config()
    )

    assert len(specs) == GLM52_LOGICAL_FACTOR_COUNT
    assert all(1 in spec.export_shape for spec in specs)


@pytest.mark.parametrize("lora_format", (None, "ordinary", "per_expert"))
def test_exact_mode_rejects_every_non_shared_outer_adapter_format(lora_format):
    adapter_config = _exact_adapter_config()
    if lora_format is None:
        adapter_config.pop("_sglang_lora_format")
    else:
        adapter_config["_sglang_lora_format"] = lora_format

    with pytest.raises(ValueError, match="requires _sglang_lora_format"):
        maybe_create_glm52_validator(_exact_official_config(), adapter_config)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("r", 2),
        ("lora_alpha", 2),
        ("lora_dropout", 0.1),
        ("bias", "all"),
        ("fan_in_fan_out", True),
        ("use_dora", True),
        ("use_rslora", True),
        ("alpha_pattern", {"lm_head": 2}),
        ("rank_pattern", {"lm_head": 2}),
    ),
)
def test_exact_inventory_rejects_adapter_program_variants(name, value):
    adapter_config = _exact_adapter_config()
    adapter_config[name] = value

    with pytest.raises(ValueError, match="exact GLM-5.2 XORL active-LoRA"):
        build_glm52_xorl_shared_outer_inventory(
            _exact_official_config(), adapter_config
        )


def test_functional_shared_outer_inventory_still_accepts_rank_two():
    specs = build_glm52_xorl_shared_outer_inventory(
        _official_config(), _adapter_config()
    )

    assert len(specs) == GLM52_LOGICAL_FACTOR_COUNT
    assert any(2 in spec.export_shape for spec in specs)


def test_inventory_has_exact_1700_raw_xorl_keys_and_role_counts():
    specs = build_glm52_xorl_shared_outer_inventory(
        _official_config(), _adapter_config()
    )
    assert len(specs) == GLM52_LOGICAL_FACTOR_COUNT
    assert len({spec.export_key for spec in specs}) == 1700
    assert {spec.export_dtype for spec in specs} == {torch.bfloat16}

    counts = summarize_glm52_factor_roles(specs)
    assert counts == {
        "attention.kv_a_proj_with_mqa": 156,
        "attention.kv_b_proj": 156,
        "attention.o_proj": 156,
        "attention.q_a_proj": 156,
        "attention.q_b_proj": 156,
        "dense_mlp.down_proj": 6,
        "dense_mlp.gate_proj": 6,
        "dense_mlp.up_proj": 6,
        "output.lm_head": 2,
        "routed_expert.down_proj": 150,
        "routed_expert.gate_proj": 150,
        "routed_expert.up_proj": 150,
        "shared_expert.down_proj": 150,
        "shared_expert.gate_proj": 150,
        "shared_expert.up_proj": 150,
    }

    key_families = {
        "attention": re.compile(
            r"base_model\.model\.model\.layers\.(\d+)\.self_attn\."
            r"(q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)\."
            r"lora_[AB]\.weight"
        ),
        "dense_mlp": re.compile(
            r"base_model\.model\.model\.layers\.([0-2])\.mlp\."
            r"(gate_proj|up_proj|down_proj)\.lora_[AB]\.weight"
        ),
        "shared_expert": re.compile(
            r"base_model\.model\.model\.layers\.(\d+)\.mlp\.shared_experts\."
            r"(gate_proj|up_proj|down_proj)\.lora_[AB]\.weight"
        ),
        "routed_expert": re.compile(
            r"base_model\.model\.model\.layers\.(\d+)\.mlp\.experts\."
            r"(w1|w2|w3)\.lora_[AB]\.weight"
        ),
        "lm_head": re.compile(r"base_model\.model\.lm_head\.lora_embedding_[AB]"),
    }
    classified = {name: [] for name in key_families}
    for spec in specs:
        matches = [
            family
            for family, pattern in key_families.items()
            if pattern.fullmatch(spec.export_key)
        ]
        assert len(matches) == 1, spec.export_key
        classified[matches[0]].append(spec)

    assert {name: len(rows) for name, rows in classified.items()} == {
        "attention": 780,
        "dense_mlp": 18,
        "shared_expert": 450,
        "routed_expert": 450,
        "lm_head": 2,
    }
    assert {spec.layer_id for spec in classified["attention"]} == set(range(78))
    assert {spec.layer_id for spec in classified["dense_mlp"]} == {0, 1, 2}
    assert {spec.layer_id for spec in classified["shared_expert"]} == set(range(3, 78))
    assert {spec.layer_id for spec in classified["routed_expert"]} == set(range(3, 78))


def test_inventory_makes_fused_slices_orientations_and_raw_names_explicit():
    specs = build_glm52_xorl_shared_outer_inventory(
        _official_config(), _adapter_config()
    )
    by_key = {spec.export_key: spec for spec in specs}
    prefix = "base_model.model.model.layers.3"

    q_a = by_key[f"{prefix}.self_attn.q_a_proj.lora_A.weight"]
    kv_a = by_key[f"{prefix}.self_attn.kv_a_proj_with_mqa.lora_A.weight"]
    assert q_a.load_key == kv_a.load_key
    assert q_a.load_key.endswith("self_attn.fused_qkv_a_proj_with_mqa.lora_A.weight")
    assert q_a.physical_slice == "rank[0:r]"
    assert kv_a.physical_slice == "rank[r:2r]"
    assert q_a.orientation == kv_a.orientation == "[rank,in]"

    gate = by_key[f"{prefix}.mlp.shared_experts.gate_proj.lora_B.weight"]
    up = by_key[f"{prefix}.mlp.shared_experts.up_proj.lora_B.weight"]
    assert gate.load_key == up.load_key
    assert gate.load_key.endswith("mlp.shared_experts.gate_up_proj.lora_B.weight")
    assert gate.physical_slice == "output[0:intermediate]"
    assert up.physical_slice == "output[intermediate:2*intermediate]"

    routed_gate_a = by_key[f"{prefix}.mlp.experts.w1.lora_A.weight"]
    routed_gate_b = by_key[f"{prefix}.mlp.experts.w1.lora_B.weight"]
    routed_down_a = by_key[f"{prefix}.mlp.experts.w2.lora_A.weight"]
    routed_down_b = by_key[f"{prefix}.mlp.experts.w2.lora_B.weight"]
    assert routed_gate_a.export_shape == (1, 2, 6144)
    assert routed_gate_b.export_shape == (256, 2048, 2)
    assert routed_down_a.export_shape == (256, 2, 2048)
    assert routed_down_b.export_shape == (1, 6144, 2)
    assert routed_gate_a.expert_layout == "shared_outer"
    assert routed_gate_b.expert_layout == "expert_local_inner"
    assert routed_down_a.expert_layout == "expert_local_inner"
    assert routed_down_b.expert_layout == "shared_outer"

    kv_b = by_key[f"{prefix}.self_attn.kv_b_proj.lora_B.weight"]
    assert kv_b.expert_layout == "absorbed_mla_correction"
    assert by_key["base_model.model.lm_head.lora_embedding_B"].export_shape == (
        154880,
        2,
    )


def test_validator_fails_closed_on_shape_missing_extra_and_target_drift():
    config = _official_config()
    adapter_config = _adapter_config()
    validator = Glm52XorlSharedOuterValidator(config, adapter_config)
    first = validator.specs[0]

    with pytest.raises(ValueError, match="has shape"):
        validator.observe(
            first.export_key, SimpleNamespace(shape=(1,), dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="has dtype"):
        validator.observe(
            first.export_key,
            SimpleNamespace(shape=first.export_shape, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="Unexpected tensor"):
        validator.observe("base_model.model.not_a_glm_factor", torch.empty(0))

    validator = Glm52XorlSharedOuterValidator(config, adapter_config)
    validator.observe(
        first.export_key,
        SimpleNamespace(shape=first.export_shape, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="missing 1699"):
        validator.finalize()

    validator = Glm52XorlSharedOuterValidator(config, adapter_config)
    for spec in validator.specs:
        validator.observe(
            spec.export_key,
            SimpleNamespace(shape=spec.export_shape, dtype=torch.bfloat16),
        )
    validator.finalize()

    bad_config = _adapter_config()
    bad_config["target_modules"] = [
        name for name in bad_config["target_modules"] if name != "kv_b_proj"
    ]
    with pytest.raises(ValueError, match="target_modules mismatch"):
        build_glm52_xorl_shared_outer_inventory(config, bad_config)

    bad_geometry_values = vars(config).copy()
    bad_geometry_values["hidden_size"] = 4096
    with pytest.raises(ValueError, match="certified GLM-5.2 adapter geometry"):
        build_glm52_xorl_shared_outer_inventory(
            SimpleNamespace(**bad_geometry_values), adapter_config
        )

    missing_frequency_values = vars(config).copy()
    missing_frequency_values.pop("moe_layer_freq")
    with pytest.raises(ValueError, match=r"config\.moe_layer_freq"):
        build_glm52_xorl_shared_outer_inventory(
            SimpleNamespace(**missing_frequency_values), adapter_config
        )


def _tensor(shape, start):
    size = 1
    for dim in shape:
        size *= dim
    return torch.arange(start, start + size, dtype=torch.float32).reshape(shape)


def test_ordinary_loader_maps_representative_xorl_tensors_to_runtime_targets():
    base_config = SimpleNamespace(
        architectures=["TinyGlmMoeDsaForCausalLM"],
        model_type="tiny_glm",
        num_hidden_layers=4,
    )
    config = LoRAConfig.from_dict(
        {
            "lora_alpha": 2,
            "peft_type": "LORA",
            "r": 2,
            "target_modules": sorted(GLM52_REQUIRED_TARGET_MODULES),
        }
    )
    adapter = LoRAAdapter(
        "generation-7",
        config,
        base_config,
        load_config=None,
        lora_backend=SimpleNamespace(),
    )

    p0 = "base_model.model.model.layers.0"
    p3 = "base_model.model.model.layers.3"
    tensors = {
        f"{p0}.self_attn.q_a_proj.lora_A.weight": _tensor((2, 8), 0),
        f"{p0}.self_attn.kv_a_proj_with_mqa.lora_A.weight": _tensor((2, 8), 100),
        f"{p0}.self_attn.q_a_proj.lora_B.weight": _tensor((4, 2), 200),
        f"{p0}.self_attn.kv_a_proj_with_mqa.lora_B.weight": _tensor((5, 2), 300),
        f"{p0}.self_attn.q_b_proj.lora_A.weight": _tensor((2, 4), 400),
        f"{p0}.self_attn.q_b_proj.lora_B.weight": _tensor((10, 2), 500),
        f"{p0}.self_attn.kv_b_proj.lora_A.weight": _tensor((2, 3), 600),
        f"{p0}.self_attn.kv_b_proj.lora_B.weight": _tensor((14, 2), 700),
        f"{p0}.self_attn.o_proj.lora_A.weight": _tensor((2, 8), 800),
        f"{p0}.self_attn.o_proj.lora_B.weight": _tensor((8, 2), 900),
        f"{p0}.mlp.gate_proj.lora_A.weight": _tensor((2, 8), 1000),
        f"{p0}.mlp.up_proj.lora_A.weight": _tensor((2, 8), 1100),
        f"{p0}.mlp.gate_proj.lora_B.weight": _tensor((16, 2), 1200),
        f"{p0}.mlp.up_proj.lora_B.weight": _tensor((16, 2), 1300),
        f"{p0}.mlp.down_proj.lora_A.weight": _tensor((2, 16), 1400),
        f"{p0}.mlp.down_proj.lora_B.weight": _tensor((8, 2), 1500),
        f"{p3}.mlp.shared_experts.gate_proj.lora_A.weight": _tensor((2, 8), 1600),
        f"{p3}.mlp.shared_experts.up_proj.lora_A.weight": _tensor((2, 8), 1700),
        f"{p3}.mlp.shared_experts.gate_proj.lora_B.weight": _tensor((6, 2), 1800),
        f"{p3}.mlp.shared_experts.up_proj.lora_B.weight": _tensor((6, 2), 1900),
        f"{p3}.mlp.shared_experts.down_proj.lora_A.weight": _tensor((2, 6), 2000),
        f"{p3}.mlp.shared_experts.down_proj.lora_B.weight": _tensor((8, 2), 2100),
        f"{p3}.mlp.experts.w1.lora_A.weight": _tensor((1, 2, 8), 2200),
        f"{p3}.mlp.experts.w1.lora_B.weight": _tensor((4, 6, 2), 2300),
        f"{p3}.mlp.experts.w2.lora_A.weight": _tensor((4, 2, 6), 2400),
        f"{p3}.mlp.experts.w2.lora_B.weight": _tensor((1, 8, 2), 2500),
        f"{p3}.mlp.experts.w3.lora_A.weight": _tensor((1, 2, 8), 2600),
        f"{p3}.mlp.experts.w3.lora_B.weight": _tensor((4, 6, 2), 2700),
        "base_model.model.lm_head.lora_embedding_A": _tensor((2, 8), 2800),
        "base_model.model.lm_head.lora_embedding_B": _tensor((11, 2), 2900),
    }
    adapter.initialize_weights_from_tensors(tensors)

    layer0 = adapter.layers[0].weights
    fused_a = f"{p0}.self_attn.fused_qkv_a_proj_with_mqa.lora_A.weight"
    fused_b = f"{p0}.self_attn.fused_qkv_a_proj_with_mqa.lora_B.weight"
    assert torch.equal(
        layer0[fused_a],
        torch.cat(
            (
                tensors[f"{p0}.self_attn.q_a_proj.lora_A.weight"],
                tensors[f"{p0}.self_attn.kv_a_proj_with_mqa.lora_A.weight"],
            ),
            dim=0,
        ),
    )
    assert torch.equal(
        layer0[fused_b],
        torch.cat(
            (
                tensors[f"{p0}.self_attn.q_a_proj.lora_B.weight"],
                tensors[f"{p0}.self_attn.kv_a_proj_with_mqa.lora_B.weight"],
            ),
            dim=0,
        ),
    )
    assert layer0[f"{p0}.mlp.gate_up_proj.lora_A.weight"].shape == (4, 8)
    assert layer0[f"{p0}.mlp.gate_up_proj.lora_B.weight"].shape == (32, 2)
    assert f"{p0}.self_attn.kv_b_proj.lora_A.weight" in layer0

    layer3 = adapter.layers[3].weights
    assert layer3[f"{p3}.mlp.shared_experts.gate_up_proj.lora_A.weight"].shape == (4, 8)
    assert layer3[f"{p3}.mlp.experts.gate_up_proj.lora_A.weight"].shape == (1, 4, 8)
    assert layer3[f"{p3}.mlp.experts.gate_up_proj.lora_B.weight"].shape == (4, 12, 2)
    assert layer3[f"{p3}.mlp.experts.down_proj.lora_A.weight"].shape == (
        4,
        2,
        6,
    )
    assert layer3[f"{p3}.mlp.experts.down_proj.lora_B.weight"].shape == (
        1,
        8,
        2,
    )
    assert set(adapter.embedding_layers) == {
        "base_model.model.lm_head.lora_embedding_A",
        "base_model.model.lm_head.lora_embedding_B",
    }


class _ForwardMode:
    def __init__(self, use_graph):
        self.use_graph = use_graph

    def is_cuda_graph(self):
        return self.use_graph

    def is_extend(self):
        return False


class _Backend:
    def __init__(self):
        self.calls = []
        self.batch_info = SimpleNamespace(has_active_lora=False)

    def prepare_lora_batch(self, **kwargs):
        self.calls.append(kwargs)


class _MemoryPool:
    def __init__(self):
        self.uid_to_buffer_id = {None: 0, "generation-6": 1, "generation-7": 2}
        self.removed = []

    def get_buffer_id(self, uid):
        return self.uid_to_buffer_id[uid]

    def remove_lora(self, uid):
        self.removed.append(uid)
        return self.uid_to_buffer_id.pop(uid, None)


def test_unique_generation_activation_graph_selection_and_unload_are_ordinary():
    manager = object.__new__(LoRAManager)
    native_fp8_base = object()
    manager.base_model = native_fp8_base
    manager.base_hf_config = SimpleNamespace(_glm52_exact_mode=False)
    manager.max_loras_per_batch = 3
    manager.max_bs_in_cuda_graph = 4
    manager.memory_pool = _MemoryPool()
    manager.lora_backend = _Backend()
    manager.loras = {
        uid: SimpleNamespace(config=SimpleNamespace(r=2), scaling=1.0)
        for uid in ("generation-6", "generation-7")
    }
    manager.can_use_prefill_cuda_graph = lambda _batch: False

    for uid, use_graph in (("generation-6", False), ("generation-7", True)):
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=_ForwardMode(use_graph),
            lora_ids=[uid],
        )
        manager.prepare_lora_batch(batch)
        call = manager.lora_backend.calls[-1]
        assert call["weight_indices"] == [manager.memory_pool.uid_to_buffer_id[uid]]
        assert call["use_cuda_graph"] is use_graph

    manager.configs = {
        uid: SimpleNamespace() for uid in ("generation-6", "generation-7")
    }
    manager.lora_refs = {
        uid: SimpleNamespace(
            lora_id=uid,
            lora_name=uid,
            lora_path=f"/adapters/{uid}",
            pinned=False,
        )
        for uid in ("generation-6", "generation-7")
    }
    manager.num_pinned_loras = 0
    manager.pending_lora_load_events = {}
    notified = []
    manager._notify_lora_slots_updated = lambda slots: notified.append(slots)

    result = manager._unload_lora_adapter(SimpleNamespace(lora_id="generation-6"))
    assert result.success
    assert manager.memory_pool.removed == ["generation-6"]
    assert notified == [{1}]
    assert set(result.loaded_adapters) == {"generation-7"}
    assert manager.base_model is native_fp8_base


def _exact_batch_manager():
    manager = object.__new__(LoRAManager)
    manager.base_hf_config = _exact_official_config()
    manager.base_model = SimpleNamespace(num_fused_shared_experts=0)
    manager.max_loras_per_batch = 3
    manager.max_bs_in_cuda_graph = 16
    manager.memory_pool = _MemoryPool()
    manager.lora_backend = _Backend()
    manager.loras = {
        "generation-6": SimpleNamespace(
            config=SimpleNamespace(r=1),
            scaling=1.0,
            _glm52_exact_adapter_certified=True,
        )
    }
    manager.can_use_prefill_cuda_graph = lambda _batch: False
    return manager


def test_exact_batch_requires_one_resident_certified_rank_one_adapter(monkeypatch):
    manager = _exact_batch_manager()
    monkeypatch.setattr(
        "sglang.srt.lora.lora_manager.get_is_capture_mode", lambda: False
    )

    active = SimpleNamespace(
        batch_size=1,
        forward_mode=_ForwardMode(True),
        lora_ids=["generation-6"],
    )
    manager.prepare_lora_batch(active)
    assert manager.lora_backend._glm52_exact_batch_certified is True
    assert manager.lora_backend.calls[-1]["weight_indices"] == [1]
    assert manager.lora_backend.calls[-1]["lora_ranks"] == [0, 1, 0]
    assert manager.lora_backend.calls[-1]["scalings"] == [0, 1.0, 0]

    for lora_ids, error in (
        (["missing"], "missing or nonresident"),
        (["generation-6", "generation-6"], "exactly one logical request"),
        ([None, "generation-6"], "exactly one logical request"),
    ):
        with pytest.raises(RuntimeError, match=error):
            manager.prepare_lora_batch(
                SimpleNamespace(
                    batch_size=len(lora_ids),
                    forward_mode=_ForwardMode(True),
                    lora_ids=lora_ids,
                )
            )
        assert manager.lora_backend._glm52_exact_batch_certified is False

    manager.loras["generation-6"]._glm52_exact_adapter_certified = False
    with pytest.raises(RuntimeError, match="complete 1,700-factor inventory"):
        manager.prepare_lora_batch(active)


def test_exact_batch_allows_only_all_base_placeholders_during_graph_capture(
    monkeypatch,
):
    manager = _exact_batch_manager()
    monkeypatch.setattr(
        "sglang.srt.lora.lora_manager.get_is_capture_mode", lambda: True
    )

    capture = SimpleNamespace(
        batch_size=16,
        forward_mode=_ForwardMode(True),
        lora_ids=[None] * 16,
    )
    manager.prepare_lora_batch(capture)
    assert manager.lora_backend._glm52_exact_batch_certified is True
    assert manager.lora_backend.calls[-1]["weight_indices"] == [0] * 16

    with pytest.raises(RuntimeError, match="synthetic base-slot placeholders"):
        manager.prepare_lora_batch(
            SimpleNamespace(
                batch_size=16,
                forward_mode=_ForwardMode(True),
                lora_ids=[None] * 15 + ["generation-6"],
            )
        )


def test_post_load_launch_contract_selects_shared_outer_triton_pool():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        [
            "--model",
            "zai-org/GLM-5.2-FP8",
            "--enable-lora",
            "--max-lora-rank",
            "2",
            "--lora-target-modules",
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "q_b_proj",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            "--experts-shared-outer-loras",
            "--disable-shared-experts-fusion",
            "--moe-runner-backend",
            "triton",
        ]
    )
    server_args = ServerArgs.from_cli_args(parsed)

    # POST loading remains ordinary, but the fixed pool geometry must be
    # selected before a shared-outer adapter is loaded into an empty server.
    assert server_args.experts_shared_outer_loras is True
    assert server_args.disable_shared_experts_fusion is True
    assert server_args.moe_runner_backend == "triton"
    assert server_args.max_lora_rank == 2
    assert set(server_args.lora_target_modules) == GLM52_REQUIRED_TARGET_MODULES
    assert get_normalized_target_modules(server_args.lora_target_modules) == {
        "down_proj",
        "fused_qkv_a_proj_with_mqa",
        "gate_up_proj",
        "kv_b_proj",
        "lm_head",
        "o_proj",
        "q_b_proj",
    }

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = _official_config()
    manager.base_model = SimpleNamespace(num_fused_shared_experts=0)
    manager.experts_shared_outer_loras = False
    lora_config = LoRAConfig.from_dict(_adapter_config())
    with pytest.raises(ValueError, match="--experts-shared-outer-loras"):
        manager._validate_glm52_runtime_layout(lora_config)

    manager.experts_shared_outer_loras = True
    manager.base_model.num_fused_shared_experts = 1
    with pytest.raises(ValueError, match="--disable-shared-experts-fusion"):
        manager._validate_glm52_runtime_layout(lora_config)

    manager.base_model.num_fused_shared_experts = 0
    manager._validate_glm52_runtime_layout(lora_config)

    manager.base_hf_config = _exact_official_config()
    ordinary_config = _exact_adapter_config()
    ordinary_config.pop("_sglang_lora_format")
    with pytest.raises(ValueError, match="ordinary or missing adapter formats"):
        manager._validate_glm52_runtime_layout(LoRAConfig.from_dict(ordinary_config))


def test_glm_model_declares_only_the_supported_complete_runtime_targets():
    assert set(GlmMoeDsaForCausalLM.supported_lora_modules) == {
        "down_proj",
        "fused_qkv_a_proj_with_mqa",
        "gate_up_proj",
        "kv_b_proj",
        "lm_head",
        "o_proj",
        "q_b_proj",
    }


class _MixedTPMergedBase(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, tp_size: int):
        super().__init__()
        assert intermediate_size % tp_size == 0
        self.tp_size = tp_size
        self.tp_rank = 0
        self.output_sizes = [intermediate_size, intermediate_size]
        self.output_partition_sizes = [intermediate_size // tp_size] * 2
        self.weight = torch.nn.Parameter(
            torch.empty(sum(self.output_partition_sizes), hidden_size)
        )


class _MixedTPRowBase(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, tp_size: int):
        super().__init__()
        assert intermediate_size % tp_size == 0
        self.tp_size = tp_size
        self.tp_rank = 0
        self.input_size_per_partition = intermediate_size // tp_size
        self.output_size = hidden_size
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, self.input_size_per_partition)
        )


def _make_manual_lora_wrapper(wrapper_type, base_layer):
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_layer = base_layer
    wrapper.lora_backend = SimpleNamespace()
    wrapper.set_lora = False
    if wrapper_type is MergedColumnParallelLinearWithLoRA:
        wrapper.n_slices = len(base_layer.output_partition_sizes)
    return wrapper


def test_memory_pool_uses_wrapped_layer_tp_for_dense_and_shared_mlp(monkeypatch):
    config = SimpleNamespace(
        first_k_dense_replace=1,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=8,
        moe_layer_freq=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_attention_heads=1,
        num_hidden_layers=2,
        vocab_size=32,
    )
    base_model = torch.nn.Module()
    base_model.config = config
    base_model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])

    dense_mlp = torch.nn.Module()
    dense_gate_up = _make_manual_lora_wrapper(
        MergedColumnParallelLinearWithLoRA,
        _MixedTPMergedBase(config.hidden_size, config.intermediate_size, tp_size=1),
    )
    dense_down = _make_manual_lora_wrapper(
        RowParallelLinearWithLoRA,
        _MixedTPRowBase(config.hidden_size, config.intermediate_size, tp_size=1),
    )
    dense_mlp.gate_up_proj = dense_gate_up
    dense_mlp.down_proj = dense_down
    base_model.layers[0].mlp = dense_mlp

    sparse_mlp = torch.nn.Module()
    sparse_mlp.shared_experts = torch.nn.Module()
    shared_gate_up = _make_manual_lora_wrapper(
        MergedColumnParallelLinearWithLoRA,
        _MixedTPMergedBase(config.hidden_size, config.moe_intermediate_size, tp_size=4),
    )
    shared_down = _make_manual_lora_wrapper(
        RowParallelLinearWithLoRA,
        _MixedTPRowBase(config.hidden_size, config.moe_intermediate_size, tp_size=4),
    )
    sparse_mlp.shared_experts.gate_up_proj = shared_gate_up
    sparse_mlp.shared_experts.down_proj = shared_down
    base_model.layers[1].mlp = sparse_mlp

    lora_modules = [
        {
            "model.layers.0.mlp.gate_up_proj": dense_gate_up,
            "model.layers.0.mlp.down_proj": dense_down,
        },
        {
            "model.layers.1.mlp.shared_experts.gate_up_proj": shared_gate_up,
            "model.layers.1.mlp.shared_experts.down_proj": shared_down,
        },
    ]
    monkeypatch.setattr(
        LoRAMemoryPool, "_has_moe_module", staticmethod(lambda _model: True)
    )
    pool = LoRAMemoryPool(
        base_hf_config=config,
        max_loras_per_batch=1,
        dtype=torch.float32,
        tp_size=4,
        tp_rank=0,
        attn_tp_size=4,
        max_lora_rank=2,
        target_modules={"gate_up_proj", "down_proj"},
        base_model=base_model,
        eviction_policy="lru",
        lora_added_tokens_size=0,
        lora_modules=lora_modules,
    )

    assert pool.B_buffer["gate_up_proj"][0].shape == (1, 24, 2)
    assert pool.B_buffer["gate_up_proj"][1].shape == (1, 4, 2)
    assert pool.A_buffer["down_proj"][0].shape == (1, 2, 12)
    assert pool.A_buffer["down_proj"][1].shape == (1, 2, 2)

    manager = object.__new__(LoRAManager)
    manager.lora_modules = lora_modules
    manager.memory_pool = pool
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    manager.update_lora_info()

    assert dense_gate_up.B_buffer.shape[-2] == sum(
        dense_gate_up.base_layer.output_partition_sizes
    )
    assert shared_gate_up.B_buffer.shape[-2] == sum(
        shared_gate_up.base_layer.output_partition_sizes
    )
    assert (
        dense_down.A_buffer.shape[-1] == dense_down.base_layer.input_size_per_partition
    )
    assert (
        shared_down.A_buffer.shape[-1]
        == shared_down.base_layer.input_size_per_partition
    )
