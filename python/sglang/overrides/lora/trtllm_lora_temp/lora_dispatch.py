"""Override twin of ``sglang.srt.lora.trtllm_lora_temp.lora_dispatch``.

Full-body replacement of the FP8 LoRA dispatch
(``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora``). The body is the
upstream-file function verbatim plus two fork behaviors; when the srt body
changes, re-derive this copy.

1. **No-adapter redirect** -- no-adapter FP8 batches go to the standard
   flashinfer-TRT-LLM FP8 implementation, the way the bf16 path already does,
   instead of into this fork's vendored ``fused_experts_fp8_sgl``: the vendored
   wrapper positionally drifted against flashinfer >= 0.6.15 (four LoRA
   parameters inserted mid-signature), and only the no-adapter branch goes
   through it. The LoRA path calls the fork's own raw op and is unaffected.

2. **FP8 gate_up LoRA delta decomposition** -- GEMM2 = W2 @ act_base +
   W2 @ act_delta, with the activation-level delta quantized per-group on its
   OWN scales (``_apply_gate_up_delta_gemm2``). The fused design quantized
   (base + delta) for GEMM2, destroying any trained delta below e4m3's
   per-element step relative to the base activation (GLM-5.2 adapters: 0.4% of
   activation -> recovered-delta cosine 0.30 fused vs 0.9998 decomposed, on
   captured real-engine tensors). Requires the split buffer support in the
   vendored kernels (``activation_lora_delta``); with the buffer unset -- EP,
   the two-stream copy, or the ``SGLANG_DISABLE_LORA_FP8_DELTA_SPLIT``
   kill-switch -- the kernel is bit-exact legacy. ``gate_up_delta`` is
   zero-initialized on the split path, which also closes a latent
   garbage-injection for mixed batches (rows of tokens without an active
   adapter were uninitialized).

Verified: Qwen3.5-35B-A3B-FP8 TP=2, eight rank-64 adapters -> 8/8 with the
split active; zero-delta split-vs-legacy output diff exactly 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def _apply_gate_up_delta_gemm2(
    *,
    activation_lora_delta,
    w2_weight,
    w2_scale,
    topk_ids,
    topk_weights,
    output,
    block_k: int,
) -> None:
    """Second half of the FP8 gate_up-delta decomposition.

    The activation kernel fed GEMM2 the DELTA-FREE activation and exported the
    activation-level LoRA delta (bf16, [num_tokens, top_k, intermediate]).
    Quantized on the base activation's scales the delta would be below e4m3's
    per-element step; quantized here on its OWN per-group amax it keeps full
    relative precision at any magnitude. One routed grouped GEMM against the
    same fp8 w2 shard, weighted by the routing weights and summed over top_k
    into the finalized output -- the same shape of work as the down-LoRA
    expand.
    """
    import triton.language as tl

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    num_tokens, top_k, intermediate = activation_lora_delta.shape
    delta_2d = activation_lora_delta.view(num_tokens * top_k, intermediate)
    a_q, a_scale = per_token_group_quant_fp8(delta_2d, block_k)
    # Static config: the GEMM is small (N = hidden per shard, K = intermediate
    # per shard) and runs once per layer; BLOCK_SIZE_K must equal the quant
    # group so each K-block reads one scale.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    # ignore_invalid_expert: CUDA-graph padded slots carry -1 expert ids; the
    # align must drop them rather than misindex the weight shard.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        w2_weight.shape[0],
        ignore_invalid_expert=True,
    )
    invoke_fused_moe_kernel(
        a_q,
        w2_weight,
        None,
        output,
        a_scale,
        w2_scale,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,  # mul_routed_weight: routing weight applied per expanded row
        1,  # top_k = 1: A rows are already the expanded (token, k) entries
        config,
        tl.bfloat16,
        True,  # use_fp8_w8a8
        False,
        False,
        False,
        False,
        block_shape=[block_k, block_k],
        fuse_sum_all_reduce=True,
        lora_preserve_base=True,
        router_topk=top_k,
    )


def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp8_block_scale_moe_lora_finalize,
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.lora_moe_runners import build_lora_hooks
    from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
        maybe_overlap_staged_shared_add,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm LoRA currently supports the gated SwiGLU FP8 "
        "Qwen path only."
    )
    assert quant_info.block_quant and not quant_info.use_mxfp8, (
        "experimental_sgl_trtllm LoRA currently supports DeepSeekFp8 block-quant "
        "checkpoints only."
    )
    assert quant_info.weight_block_k is not None
    assert quant_info.w13_weight_scale_inv is not None
    assert quant_info.w2_weight_scale_inv is not None

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    if not get_is_capture_mode() and not lora_info.has_active_lora:
        # Twin redirect (see module docstring): the vendored no-adapter wrapper
        # positionally drifted against flashinfer >= 0.6.15; route to the
        # upstream implementation instead.
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            fused_experts_none_to_flashinfer_trtllm_fp8,
        )

        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if use_virtual_lora_store:
        hooks = None
        token_lora_mapping = lora_info.token_lora_mapping
        fused_lora_routing_cache: dict = {}
    else:
        hooks = build_lora_hooks(hidden_states, lora_info, topk_ids)
        token_lora_mapping = None
        fused_lora_routing_cache = {}

    # Fuse the per-token scale transpose into the quant kernel (column-major scales) so the
    # `.t()` is a free view -> drops the standalone ~2us transpose+copy. Byte/shape-identical.
    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    # EP-aware LoRA: under MoE EP each rank computes the delta only for the experts it
    # owns (passed via local_expert_offset/local_num_experts below). gate_up_delta stays
    # new_empty even though non-owned [token, k] slots are then left unwritten -- the
    # trtllm MoE is itself EP-aware, so those slots never feed the all-reduced output.
    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    # Gate_up delta decomposition: GEMM2's quantized operand stays delta-free
    # and the activation-level delta runs through its own own-scale FP8 GEMM
    # (_apply_gate_up_delta_gemm2). A trained delta below e4m3's per-element
    # step relative to the base activation is otherwise replaced by
    # quantization-grid noise (GLM-5.2 adapters: 0.4% of activation ->
    # recovered-delta cosine 0.30 fused vs 0.9998 decomposed). Global expert
    # ids index the local w2 shard in the extra GEMM, so EP stays on the
    # legacy fused path.
    use_split_gate_up_delta = (
        use_virtual_lora_store
        and quant_info.local_num_experts == quant_info.global_num_experts
        and not envs.SGLANG_DISABLE_LORA_FP8_DELTA_SPLIT.get()
    )
    gate_up_delta = (
        # Zero-init when split: the activation kernel derives the exported
        # delta from this buffer for EVERY token, including ones without an
        # active adapter, whose rows the merged add below never writes.
        hidden_states.new_zeros(gate_up_delta_shape)
        if use_split_gate_up_delta or not use_virtual_lora_store
        else hidden_states.new_empty(gate_up_delta_shape)
    )
    activation_lora_delta = (
        torch.empty(
            (
                hidden_states.shape[0],
                runner_config.top_k,
                quant_info.intermediate_size,
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if use_split_gate_up_delta
        else None
    )
    if use_virtual_lora_store:
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
        )
    elif hooks.after_gate_up is not None:
        hooks.after_gate_up(hidden_states, gate_up_delta, topk_weights, topk_ids)

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # SGLANG_OPT_LORA_FUSED_TOPK_PACK: the routed pack may already have been produced
    # fused inside the gating kernel (StandardTopKOutput.packed_topk_ids) — including
    # the padded-region id=-1 mask. Fall back to the separate pack otherwise.
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    direct_down_output = None
    if use_virtual_lora_store:
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            direct_down_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

    moe_result = trtllm_fp8_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale_inv,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale_inv,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=(
            RoutingMethodType.TopK
            if quant_info.routing_method_type == RoutingMethodType.DeepSeekV3
            else quant_info.routing_method_type
        ),
        use_shuffled_weight=False,
        do_finalize=use_virtual_lora_store,
        output=(
            direct_down_output
            if direct_down_output is not None
            else torch.empty_like(hidden_states)
        ),
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
        activation_lora_delta=activation_lora_delta,
    )
    if use_virtual_lora_store:
        output = moe_result
        # Shared-add overlap: the trtllm op above already finalized `output`, so the
        # staged shared-expert add (if any) can run on the main stream concurrent with
        # the down-LoRA shrink below; the expand waits on it via expand_wait_event.
        shared_add_done = maybe_overlap_staged_shared_add(output)
        merged_experts_fused_moe_lora_add(
            output=output,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            expand_wait_event=shared_add_done,
        )
        if activation_lora_delta is not None:
            _apply_gate_up_delta_gemm2(
                activation_lora_delta=activation_lora_delta,
                w2_weight=quant_info.w2_weight,
                w2_scale=quant_info.w2_weight_scale_inv,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=output,
                block_k=quant_info.weight_block_k,
            )
        return StandardCombineInput(hidden_states=output)

    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = moe_result

    down_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        hidden_states.shape[1],
    )
    down_delta = (
        hidden_states.new_empty(down_delta_shape)
        if use_virtual_lora_store
        else hidden_states.new_zeros(down_delta_shape)
    )
    if use_virtual_lora_store:
        merged_experts_fused_moe_lora_add(
            output=down_delta,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
        )
    elif hooks.after_down is not None:
        hooks.after_down(
            activation_lora_input.view(-1, quant_info.intermediate_size),
            down_delta,
            topk_weights,
            topk_ids,
        )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    output = trtllm_fp8_block_scale_moe_lora_finalize(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        down_lora_delta=down_delta,
        output=output,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )

    return StandardCombineInput(hidden_states=output)


def __apply_patch__(public_mod):
    public_mod.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora
    )
    # Exposed for offline replay/validation scripts.
    public_mod._apply_gate_up_delta_gemm2 = _apply_gate_up_delta_gemm2
