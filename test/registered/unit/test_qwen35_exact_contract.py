"""Conventional contract tests for Qwen3.5-family exact serving."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from sglang.kernels.ops.attention.fla import qwen35_gdn_exact as exact
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
)
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
    args = _server_args(tp_size=8, ep_size=1, disable_cuda_graph_padding=True)
    # Match the real __post_init__ ordering: generic graph resolution has
    # already materialized defaults before model-specific contracts run.
    args.cuda_graph_config = default_cuda_graph_config()
    args._cuda_graph_config_locked = set()

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
    assert args.max_mamba_cache_size == 1280
    assert args.cuda_graph_bs_decode == list(range(1, 33))
    assert args.cuda_graph_max_bs_decode == 32
    assert args.cuda_graph_config.decode.bs == list(range(1, 33))
    assert args.cuda_graph_config.decode.max_bs == 32
    assert (Phase.DECODE, "bs") in args._cuda_graph_config_locked
    assert (Phase.DECODE, "max_bs") in args._cuda_graph_config_locked
    assert args.disable_prefill_cuda_graph
    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED
    assert (Phase.PREFILL, "backend") in args._cuda_graph_config_locked
    assert args.disable_overlap_schedule
    assert args.disable_cuda_graph_padding
    assert args.max_running_requests == 256
    assert args.max_queued_requests == 512
    assert args.chunked_prefill_size == -1
    assert args.max_prefill_tokens == 32768
    assert args.mem_fraction_static == 0.40
    assert not args.disable_cuda_graph
    assert args.disable_radix_cache


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
        ("max_running_requests", 16, "max-running-requests 256"),
        ("mem_fraction_static", 0.5, "mem-fraction-static 0.40"),
        ("max_mamba_cache_size", 512, "max-mamba-cache-size 1280"),
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


def test_qwen35_moe_rejects_explicit_non_graph32_program():
    hf_config = _qwen_config(moe=True)
    args = _server_args(tp_size=8, cuda_graph_bs_decode=[8, 10])
    with pytest.raises(ValueError, match="graph buckets through 32"):
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

    active_flags = [
        (prefill, "BI_GDN_PREFILL_ENABLED"),
        (decode, "BI_GDN_DECODE_ENABLED"),
        (decode, "BI_GDN_BS1_STATIC"),
        (decode, "BI_GDN_DECODE_GRAPH"),
        (fast, "BI_GDN_DECODE_FAST_ENABLED"),
    ]
    held_flags = [
        (prefill, "BI_GDN_SOLVE_TRIL_DECODE"),
        (fast, "BI_GDN_FUSE_SMALL_ENABLED"),
        (incremental, "BI_GDN_DECODE_INCR_ENABLED"),
        (incremental, "BI_GDN_INCR_DEFER_ENABLED"),
        (incremental, "BI_GDN_VNEW_SLIM_ENABLED"),
        (heal, "BI_GDN_LAZY_HEAL_ENABLED"),
    ]
    with ExitStack() as stack:
        stack.enter_context(patch.object(exact, "_applied", False))
        stack.enter_context(patch.object(bi_ops, "ENABLE_JIT_DEEPGEMM", True))
        for module, name in active_flags + held_flags:
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

        assert all(getattr(module, name) for module, name in active_flags)
        assert not any(getattr(module, name) for module, name in held_flags)
        force_table.assert_called_once_with(True)
        set_norm.assert_called_once_with(4)
        set_tiera.assert_called_once_with(False)
        set_router.assert_called_once_with(False)
        set_head.assert_called_once_with(False)
        set_combine.assert_called_once_with(False)
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


def test_gdn_backend_reads_architecture_resolver_flags_at_call_time():
    import sglang.srt.layers.attention.linear.gdn_backend as backend

    with (
        patch.object(backend, "is_cuda", return_value=True),
        patch.object(backend._bi_decode_mod, "BI_GDN_DECODE_ENABLED", True),
        patch.object(backend._bi_fast_mod, "BI_GDN_DECODE_FAST_ENABLED", True),
        patch.object(backend._bi_prefill_mod, "BI_GDN_PREFILL_ENABLED", True),
    ):
        assert backend._bi_gdn_decode_enabled()
        assert backend._bi_gdn_decode_fast_enabled()
        assert backend._bi_gdn_prefill_enabled()


def test_qwen35_gemma_norm_routes_the_certified_family_split():
    import sglang.srt.layers.layernorm as layernorm

    norm = layernorm.GemmaRMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16)
    sentinel = torch.full_like(x, 7.0)

    with (
        patch.object(layernorm, "is_batch_invariant_mode_enabled", return_value=True),
        patch.object(layernorm, "is_batch_invariant_op_enabled", return_value=True),
        patch.object(
            layernorm, "rms_norm_batch_invariant", return_value=sentinel.float()
        ) as family_one,
        patch.object(norm, "forward_native", wraps=norm.forward_native) as family_two,
    ):
        assert torch.equal(norm.forward_cuda(x), sentinel)
        family_one.assert_called_once()
        args = family_one.call_args.args
        assert args[0].dtype == torch.float32
        assert torch.equal(args[1], 1.0 + norm.weight.data.float())
        family_two.assert_not_called()

        out, residual_out = norm.forward_cuda(x, residual)
        family_two.assert_called_once_with(x, residual, None)
        assert out.dtype == torch.bfloat16
        assert torch.equal(residual_out, x + residual)


def _exact_logits_processor(vocab_size=32):
    from sglang.srt.layers.logits_processor import LogitsProcessor

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.vocab_size = vocab_size
    processor.use_qwen35_bi_lm_head = True
    processor.use_fp32_lm_head = True
    processor.do_tensor_parallel_all_gather = False
    processor.do_tensor_parallel_all_gather_dp_attn = False
    processor.logit_scale = None
    processor.final_logit_softcapping = None
    return processor


def test_qwen35_decode_lm_head_calls_the_contract_gemm():
    import sglang.srt.batch_invariant_ops as bi_ops

    processor = _exact_logits_processor()
    hidden = torch.zeros((3, 8), dtype=torch.bfloat16)
    weight = torch.zeros((32, 8), dtype=torch.bfloat16)
    expected = torch.full((3, 32), 7.0, dtype=torch.float32)

    with patch.object(
        bi_ops, "bi_lm_head_full_logits", return_value=expected
    ) as contract_head:
        actual = processor._bi_lm_head_next_token_logits(
            hidden, SimpleNamespace(weight=weight), SimpleNamespace()
        )

    assert actual is expected
    contract_head.assert_called_once()
    assert torch.equal(contract_head.call_args.args[0], hidden)
    assert contract_head.call_args.args[1].data_ptr() == weight.data_ptr()


def test_qwen35_input_logprobs_call_the_contract_rescore():
    import sglang.srt.batch_invariant_ops as bi_ops

    processor = _exact_logits_processor()
    hidden = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    indices = torch.tensor([0, 2, 3])
    token_ids = torch.tensor([2, 4, 6])
    weight = torch.zeros((32, 8), dtype=torch.bfloat16)
    expected = torch.tensor([-1.0, -2.0, -3.0])

    with patch.object(
        bi_ops,
        "bi_lm_head_selected_logprob",
        return_value=(expected, None, None),
    ) as contract_rescore:
        actual = processor._bi_lm_head_input_token_logprobs(
            hidden,
            indices,
            SimpleNamespace(weight=weight),
            SimpleNamespace(extend_input_logprob_token_ids_gpu=token_ids),
        )

    assert actual is expected
    args = contract_rescore.call_args.args
    assert torch.equal(args[0], hidden[indices])
    assert args[1].data_ptr() == weight.data_ptr()
    assert args[2] is token_ids
    assert contract_rescore.call_args.kwargs == {"temperature": None}


def test_qwen35_prefill_routes_both_outputs_without_stock_lm_head():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    processor = _exact_logits_processor()
    pruned = torch.zeros((4, 8), dtype=torch.bfloat16)
    sample_indices = torch.tensor([1, 3])
    input_indices = torch.tensor([0, 2])
    exact_input = torch.tensor([-1.0, -2.0])
    exact_sample = torch.zeros((2, 32), dtype=torch.float32)
    processor._get_pruned_states = MagicMock(
        return_value=(
            pruned,
            None,
            None,
            sample_indices,
            input_indices,
            [0, 0, 1, 1],
        )
    )
    processor._get_hidden_states_to_store = MagicMock(return_value=None)
    processor._bi_lm_head_input_token_logprobs = MagicMock(return_value=exact_input)
    processor._bi_lm_head_next_token_logits = MagicMock(return_value=exact_sample)
    processor._bi_lm_head_decode_active = MagicMock(return_value=True)
    processor.input_logprob_processor = MagicMock()
    metadata = SimpleNamespace(
        forward_mode=SimpleNamespace(is_dllm_extend=lambda: False),
        extend_return_logprob=True,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        mm_input_embeds=None,
    )

    output = processor.forward(
        input_ids=torch.tensor([1]),
        hidden_states=torch.zeros((4, 8), dtype=torch.bfloat16),
        lm_head=SimpleNamespace(weight=torch.zeros((32, 8), dtype=torch.bfloat16)),
        logits_metadata=metadata,
    )

    assert isinstance(output, LogitsProcessorOutput)
    assert output.input_token_logprobs is exact_input
    assert output.next_token_logits is exact_sample
    processor.input_logprob_processor.forward.assert_not_called()
    assert torch.equal(
        processor._bi_lm_head_next_token_logits.call_args.args[0],
        pruned[sample_indices],
    )


def _exact_sampling_info(n=2, **overrides):
    values = dict(
        temperatures=torch.ones((n, 1), dtype=torch.float32),
        return_sampling_masks=[],
        is_all_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        grammar_mask=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen35_sampler_contract_rescore_uses_temperature_and_fast_head():
    import sglang.srt.layers.sampler as sampler_module

    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.use_ascend_backend = False
    sampler.return_original_logprob = False
    sampler.use_log_softmax_logprob = True
    sampler.enable_deterministic = True
    logits = torch.zeros((2, 32), dtype=torch.float32)
    token_ids = torch.tensor([3, 5])
    temperatures = torch.tensor([[0.7], [1.3]])
    expected = torch.tensor([-0.5, -0.75])

    with (
        patch.object(sampler_module, "is_bi_head_fastpath_enabled", return_value=True),
        patch(
            "sglang.srt.batch_invariant_ops.bi_head_fastpath."
            "bi_lm_head_selected_logprob_from_logits_fast",
            return_value=(expected, None, None),
        ) as fast_rescore,
    ):
        actual = sampler._bi_contract_sampled_logprob(
            logits,
            token_ids,
            _exact_sampling_info(temperatures=temperatures),
        )

    assert actual is expected
    fast_rescore.assert_called_once()
    args = fast_rescore.call_args.args
    assert args[0] is logits
    assert args[1] is token_ids
    assert torch.equal(
        fast_rescore.call_args.kwargs["temperature"], temperatures.reshape(-1)
    )


def test_qwen35_sampler_forward_overwrites_stock_logprob_with_contract_value():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    import sglang.srt.layers.sampler as sampler_module

    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.return_original_logprob = False
    sampler._glm52_exact_mode = False
    sampler.use_qwen35_bi_decode_rescore = True
    sampler.output_logprob_processor = MagicMock()
    stock = MagicMock()
    stock.write_output_to.side_effect = lambda output: setattr(
        output, "next_token_logprobs", torch.tensor([-9.0, -9.0])
    )
    sampler.output_logprob_processor.compute_logprobs.return_value = stock
    sampler._preprocess_logits = MagicMock(side_effect=lambda logits, _: logits)
    expected = torch.tensor([-0.25, -0.5])
    sampler._bi_contract_sampled_logprob = MagicMock(return_value=expected)
    sampler._sync_token_ids_across_tp = MagicMock()
    output = LogitsProcessorOutput(
        next_token_logits=torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    )

    token_ids = sampler.forward(
        output,
        _exact_sampling_info(is_all_greedy=True),
        return_logprob=True,
        top_logprobs_nums=[0, 0],
        token_ids_logprobs=[None, None],
        positions=torch.tensor([0, 0]),
    )

    assert torch.equal(token_ids, torch.tensor([0, 1]))
    assert output.next_token_logprobs is expected
    sampler._bi_contract_sampled_logprob.assert_called_once()


def test_qwen35_sampler_rescore_fails_closed_after_logit_mutation():
    import sglang.srt.layers.sampler as sampler_module

    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.use_ascend_backend = False
    sampler.return_original_logprob = False
    sampler.use_log_softmax_logprob = True
    sampler.enable_deterministic = True

    with pytest.raises(ValueError, match="does not support logit bias"):
        sampler._bi_contract_sampled_logprob(
            torch.zeros((1, 32), dtype=torch.float32),
            torch.tensor([0]),
            _exact_sampling_info(n=1, logit_bias=torch.zeros((1, 32))),
        )
