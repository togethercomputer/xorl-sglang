"""Conventional contract tests for Qwen3.5-family exact serving."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.kernels.ops.attention.fla import qwen35_gdn_exact as exact
from sglang.srt.server_args import ServerArgs


def _server_args(**overrides):
    args = ServerArgs(model_path="dummy")
    args.rl_on_policy_target = "xorl"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _qwen_config(*, moe: bool):
    layers = 40 if moe else 24
    text_config = SimpleNamespace(
        hidden_size=2048 if moe else 1024,
        num_hidden_layers=layers,
        num_attention_heads=16 if moe else 8,
        num_key_value_heads=2,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=32 if moe else 16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )
    if moe:
        text_config.num_experts = 256
        text_config.num_experts_per_tok = 8
    return SimpleNamespace(text_config=text_config)


def test_qwen35_moe_plain_config_resolves_certified_topology():
    hf_config = _qwen_config(moe=True)
    args = _server_args(tp_size=8, ep_size=1)

    args._resolve_qwen35_gdn_exact_contract(
        hf_config, model_arch="Qwen3_5MoeForConditionalGeneration"
    )

    assert args.qwen35_gdn_exact_mode
    assert hf_config._qwen35_gdn_exact_mode
    assert args.dtype == "bfloat16"
    assert args.attention_backend == "fa4"
    assert args.linear_attn_prefill_backend == "triton"
    assert args.linear_attn_decode_backend == "triton"
    assert args.enable_fp32_lm_head
    assert args.enable_fp32_router
    assert args.enable_deterministic_inference
    assert args.sampling_backend == "pytorch"
    assert args.sampling_defaults == "openai"
    assert args.disable_custom_all_reduce
    assert args.tp_size == args.dp_size == args.ep_size == 8
    assert args.pp_size == 1
    assert args.enable_dp_attention
    assert args.enable_dp_lm_head
    assert args.disable_piecewise_cuda_graph
    assert args.max_mamba_cache_size == 1024
    assert args.cuda_graph_bs_decode == [10]
    assert args.cuda_graph_max_bs_decode == 10
    assert args.disable_cuda_graph_padding
    assert args.max_running_requests == 80
    assert args.max_queued_requests == 512
    assert args.chunked_prefill_size == 16384
    assert args.max_prefill_tokens == 32768
    assert args.mem_fraction_static == 0.40
    assert not args.disable_cuda_graph
    assert not args.disable_radix_cache


def test_qwen35_dense_resolves_only_the_certified_single_rank_topology():
    hf_config = _qwen_config(moe=False)
    args = _server_args(tp_size=1, dp_size=1, ep_size=1, pp_size=1)

    args._resolve_qwen35_gdn_exact_contract(
        hf_config, model_arch="Qwen3_5ForConditionalGeneration"
    )

    assert args.qwen35_gdn_exact_mode
    assert args.tp_size == 1
    assert args.ep_size == 1
    assert not args.enable_dp_attention
    assert not args.enable_dp_lm_head
    assert not args.enable_fp32_router
    assert args.enable_deterministic_inference
    assert args.sampling_backend == "pytorch"
    assert args.sampling_defaults == "openai"
    assert args.disable_piecewise_cuda_graph
    assert args.disable_cuda_graph
    assert args.disable_radix_cache


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("tp_size", 4, "TP8/DP8"),
        ("dp_size", 4, "DP8"),
        ("ep_size", 4, "EP8"),
        ("pp_size", 2, "PP1"),
        ("dtype", "float16", "BF16"),
        ("quantization", "fp8", "unquantized"),
        ("attention_backend", "triton", "FA4"),
        ("linear_attn_prefill_backend", "flashinfer", "triton"),
        ("moe_a2a_backend", "deepep", "moe-a2a-backend none"),
        ("speculative_algorithm", "EAGLE", "speculative"),
        ("disable_cuda_graph", True, "CUDA graph bucket"),
        ("disable_radix_cache", True, "requires radix cache"),
        ("max_running_requests", 16, "max-running-requests 80"),
        ("mem_fraction_static", 0.5, "mem-fraction-static 0.40"),
        ("max_mamba_cache_size", 512, "max-mamba-cache-size 1024"),
    ],
)
def test_qwen35_moe_rejects_programs_outside_the_certified_envelope(
    name, value, message
):
    hf_config = _qwen_config(moe=True)
    args = _server_args(tp_size=8, ep_size=1)
    setattr(args, name, value)
    if name == "mem_fraction_static":
        args._mem_fraction_static_user_supplied = True

    with pytest.raises(ValueError, match=message):
        args._resolve_qwen35_gdn_exact_contract(
            hf_config, model_arch="Qwen3_5MoeForConditionalGeneration"
        )


def test_non_qwen_or_non_xorl_does_not_resolve_exact_mode():
    hf_config = SimpleNamespace()
    non_qwen = _server_args()
    non_qwen._resolve_qwen35_gdn_exact_contract(
        hf_config, model_arch="LlamaForCausalLM"
    )
    assert not non_qwen.qwen35_gdn_exact_mode

    non_xorl = _server_args(rl_on_policy_target=None)
    non_xorl._resolve_qwen35_gdn_exact_contract(
        hf_config, model_arch="Qwen3_5MoeForConditionalGeneration"
    )
    assert not non_xorl.qwen35_gdn_exact_mode


@pytest.mark.parametrize("name", ("tp_size", "dp_size", "ep_size", "pp_size"))
def test_qwen35_dense_rejects_unqualified_distributed_topologies(name):
    hf_config = _qwen_config(moe=False)
    args = _server_args()
    setattr(args, name, 2)
    with pytest.raises(ValueError, match="TP1/DP1/EP1/PP1"):
        args._resolve_qwen35_gdn_exact_contract(
            hf_config, model_arch="Qwen3_5ForConditionalGeneration"
        )


def test_qwen35_moe_rejects_explicit_non_graph10_program():
    hf_config = _qwen_config(moe=True)
    args = _server_args(tp_size=8, cuda_graph_bs_decode=[8, 10])
    with pytest.raises(ValueError, match="cuda-graph-bs 10"):
        args._resolve_qwen35_gdn_exact_contract(
            hf_config, model_arch="Qwen3_5MoeForConditionalGeneration"
        )


@pytest.mark.parametrize("moe", (False, True))
def test_qwen35_rejects_architecture_alias_with_unqualified_geometry(moe):
    hf_config = _qwen_config(moe=moe)
    hf_config.text_config.hidden_size += 1
    args = _server_args(tp_size=8 if moe else 1)
    architecture = (
        "Qwen3_5MoeForConditionalGeneration"
        if moe
        else "Qwen3_5ForConditionalGeneration"
    )
    with pytest.raises(ValueError, match="qualified model geometry.*hidden_size"):
        args._resolve_qwen35_gdn_exact_contract(hf_config, model_arch=architecture)


def test_qwen35_private_resolver_installs_one_tuple_once():
    import sglang.srt.batch_invariant_ops.batch_invariant_ops as bi_ops
    import sglang.srt.batch_invariant_ops.bi_gemm_configs as gemm_configs
    import sglang.srt.batch_invariant_ops.bi_gemm_tiera as tiera
    import sglang.srt.distributed.communication_op as communication
    import sglang.kernels.ops.attention.fla.bi_gdn_decode as decode
    import sglang.kernels.ops.attention.fla.bi_gdn_decode_fast as fast
    import sglang.kernels.ops.attention.fla.bi_gdn_decode_incr as incremental
    import sglang.kernels.ops.attention.fla.bi_gdn_incr_lazy_heal as heal
    import sglang.kernels.ops.attention.fla.bi_gdn_prefill as prefill
    import sglang.kernels.ops.attention.fla.layernorm_gated as norm
    import sglang.srt.layers.xorl_batch_invariant as xorl_family

    direct_flags = [
        (prefill, "BI_GDN_PREFILL_ENABLED"),
        (prefill, "BI_GDN_SOLVE_TRIL_DECODE"),
        (decode, "BI_GDN_DECODE_ENABLED"),
        (decode, "BI_GDN_BS1_STATIC"),
        (decode, "BI_GDN_DECODE_GRAPH"),
        (fast, "BI_GDN_DECODE_FAST_ENABLED"),
        (fast, "BI_GDN_FUSE_SMALL_ENABLED"),
        (incremental, "BI_GDN_DECODE_INCR_ENABLED"),
        (incremental, "BI_GDN_INCR_DEFER_ENABLED"),
        (incremental, "BI_GDN_VNEW_SLIM_ENABLED"),
        (heal, "BI_GDN_LAZY_HEAL_ENABLED"),
    ]
    with ExitStack() as stack:
        stack.enter_context(patch.object(exact, "_applied", False))
        stack.enter_context(patch.object(bi_ops, "ENABLE_JIT_DEEPGEMM", True))
        for module, name in direct_flags:
            stack.enter_context(patch.object(module, name, False))
        force_table = stack.enter_context(
            patch.object(gemm_configs, "_force_bi_gemm_config_table")
        )
        set_norm = stack.enter_context(
            patch.object(norm, "set_gdn_norm_rows_per_block_pin")
        )
        set_tiera = stack.enter_context(patch.object(tiera, "set_tiera_enabled"))
        set_router = stack.enter_context(
            patch.object(bi_ops, "set_router_renorm_fused_enabled")
        )
        set_head = stack.enter_context(
            patch.object(bi_ops, "set_bi_head_fastpath_enabled")
        )
        set_combine = stack.enter_context(
            patch.object(communication, "set_ordered_combine_fused_enabled")
        )
        startup = stack.enter_context(patch.object(exact.logger, "info"))
        force_family = stack.enter_context(
            patch.object(xorl_family, "force_xorl_bi_family")
        )

        server_args = SimpleNamespace(
            disable_cuda_graph=False,
            qwen35_gdn_exact_is_moe=True,
        )
        exact._apply_qwen35_gdn_exact(server_args)
        exact._apply_qwen35_gdn_exact(server_args)

        assert all(getattr(module, name) for module, name in direct_flags)
        force_table.assert_called_once_with(True)
        set_norm.assert_called_once_with(4)
        set_tiera.assert_called_once_with(True)
        set_router.assert_called_once_with(True)
        set_head.assert_called_once_with(True)
        set_combine.assert_called_once_with(True)
        startup.assert_called_once()
        force_family.assert_not_called()


def test_qwen35_public_module_has_no_partial_selection_api():
    assert "_apply_qwen35_gdn_exact" not in exact.__all__
    assert "Qwen3_6ForCausalLM" not in exact.QWEN35_EXACT_ARCHS


def test_dense_tuple_uses_only_the_directly_certified_conservative_stack():
    import sglang.srt.batch_invariant_ops.batch_invariant_ops as bi_ops
    import sglang.srt.batch_invariant_ops.bi_gemm_configs as gemm_configs
    import sglang.srt.batch_invariant_ops.bi_gemm_tiera as tiera
    import sglang.srt.distributed.communication_op as communication
    import sglang.kernels.ops.attention.fla.bi_gdn_decode as decode
    import sglang.kernels.ops.attention.fla.bi_gdn_decode_fast as fast
    import sglang.kernels.ops.attention.fla.bi_gdn_decode_incr as incremental
    import sglang.kernels.ops.attention.fla.bi_gdn_incr_lazy_heal as heal
    import sglang.kernels.ops.attention.fla.bi_gdn_prefill as prefill

    with (
        patch.object(exact, "_applied", False),
        patch.object(bi_ops, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(prefill, "BI_GDN_SOLVE_TRIL_DECODE", True),
        patch.object(decode, "BI_GDN_BS1_STATIC", True),
        patch.object(decode, "BI_GDN_DECODE_GRAPH", True),
        patch.object(fast, "BI_GDN_DECODE_FAST_ENABLED", True),
        patch.object(fast, "BI_GDN_FUSE_SMALL_ENABLED", True),
        patch.object(incremental, "BI_GDN_DECODE_INCR_ENABLED", True),
        patch.object(incremental, "BI_GDN_INCR_DEFER_ENABLED", True),
        patch.object(incremental, "BI_GDN_VNEW_SLIM_ENABLED", True),
        patch.object(heal, "BI_GDN_LAZY_HEAL_ENABLED", True),
        patch.object(gemm_configs, "_force_bi_gemm_config_table") as table,
        patch.object(tiera, "set_tiera_enabled") as tier_a,
        patch.object(bi_ops, "set_router_renorm_fused_enabled") as router,
        patch.object(bi_ops, "set_bi_head_fastpath_enabled") as head,
        patch.object(communication, "set_ordered_combine_fused_enabled") as combine,
    ):
        exact._apply_qwen35_gdn_exact(
            SimpleNamespace(
                disable_cuda_graph=True,
                qwen35_gdn_exact_is_moe=False,
            )
        )
        assert not prefill.BI_GDN_SOLVE_TRIL_DECODE
        assert not decode.BI_GDN_BS1_STATIC
        assert not decode.BI_GDN_DECODE_GRAPH
        assert not fast.BI_GDN_DECODE_FAST_ENABLED
        assert not fast.BI_GDN_FUSE_SMALL_ENABLED
        assert not incremental.BI_GDN_DECODE_INCR_ENABLED
        assert not incremental.BI_GDN_INCR_DEFER_ENABLED
        assert not incremental.BI_GDN_VNEW_SLIM_ENABLED
        assert not heal.BI_GDN_LAZY_HEAL_ENABLED
        table.assert_called_once_with(False)
        tier_a.assert_called_once_with(False)
        router.assert_called_once_with(False)
        head.assert_called_once_with(False)
        combine.assert_called_once_with(False)


def test_model_runner_resolves_qwen_exact_bi_ops_before_construction():
    from sglang.srt.server_args import _exact_batch_invariant_ops

    qwen = SimpleNamespace(qwen35_gdn_exact_mode=True, glm52_exact_mode=False)
    unrelated = SimpleNamespace(
        qwen35_gdn_exact_mode=False,
        glm52_exact_mode=False,
        enable_deterministic_inference=True,
    )
    assert _exact_batch_invariant_ops(qwen) == exact.QWEN35_REQUIRED_BI_OPS
    assert _exact_batch_invariant_ops(unrelated) is None
