"""Marlin MoE runner core with hook support for LoRA injection.

Uses Marlin int4/int8 kernels for the base MoE projections.
LoRA deltas are injected via hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        moe_sum_reduce_triton,
    )
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type,
        select_marlin_moe_block_size_m,
        swiglu_limit_func,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        moe_align_block_size,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace


class MarlinLoraRunnerCore:
    """
    MoE runner using Marlin kernels for base projections, with hooks for LoRA.

    Pipeline:
      1. moe_wna16_marlin_gemm (gate_up)
      1.5. hooks.after_gate_up
      2. silu_and_mul
      3. moe_wna16_marlin_gemm (down)
      3.5. hooks.after_down
      4. moe_sum_reduce
    """

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MarlinMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks=None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        assert hooks is not None, "hooks must be provided for MarlinLoraRunnerCore"

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        assert runner_config.activation == "silu", "Only SiLU activation is supported."
        assert (
            torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        ), "MarlinLoraRunnerCore requires CUDA compute capability >= 9"
        routed_scaling_factor = runner_config.routed_scaling_factor

        M, K = hidden_states.shape
        E = quant_info.w13_qweight.shape[0]
        N = quant_info.w2_qweight.shape[1] * 16
        topk = topk_ids.shape[1]
        num_bits = quant_info.weight_bits
        is_mxfp4_marlin = (
            num_bits == 4
            and quant_info.w13_qzeros is None
            and quant_info.w2_qzeros is None
            and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
            and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        )
        use_atomic_add = (
            hidden_states.dtype == torch.float16
            or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        ) and not is_mxfp4_marlin

        block_size_m = select_marlin_moe_block_size_m(
            num_tokens=M,
            topk=topk,
            local_experts=E,
            global_experts=(
                quant_info.global_num_experts
                if quant_info.global_num_experts != -1
                else E
            ),
            hidden_size=K,
            intermediate_size=N,
            is_mxfp4_marlin=is_mxfp4_marlin,
            clamp_limit=runner_config.swiglu_limit,
        )

        # Under EP the dispatcher already localized topk_ids (-1 = non-local); align
        # over the global expert count like fused_marlin_moe, not the local E.
        align_num_experts = (
            quant_info.global_num_experts if quant_info.expert_map is not None else E
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, align_num_experts
        )

        # Per-call workspace like fused_experts_none_to_marlin: a shared buffer aliases
        # inter-block locks across in-flight kernels/graphs and deadlocks capture.
        workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

        # Pass scales + global scale so fp4-marlin weights (the ModelOpt NVFP4
        # W4A16 fallback) resolve to float4_e2m1f instead of uint4b8.
        scalar_type1 = get_scalar_type(
            num_bits,
            quant_info.w13_qzeros is not None,
            quant_info.w13_scales,
            quant_info.w13_global_scale,
        )
        scalar_type2 = get_scalar_type(
            num_bits,
            quant_info.w2_qzeros is not None,
            quant_info.w2_scales,
            quant_info.w2_global_scale,
        )

        # Match fused_marlin_moe's cache ownership exactly.  Under EP, Marlin
        # skips every -1 (non-local) route, so those rows must start at zero;
        # exposing uninitialized rows to the LoRA hook/activation is both
        # nondeterministic and unsafe.  The shared zero allocation also keeps
        # gate/up and down cache layout identical to the base implementation.
        intermediate_cache2 = torch.empty(
            (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        intermediate_cache13 = torch.zeros(
            (M * topk * max(2 * N, K),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = intermediate_cache13[: M * topk * 2 * N].view(
            M * topk, 2 * N
        )
        intermediate_cache3 = intermediate_cache13[: M * topk * K].view(M * topk, K)
        intermediate_cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            intermediate_cache1,
            quant_info.w13_qweight,
            quant_info.w13_bias,
            quant_info.w13_scales,
            quant_info.w13_global_scale,
            quant_info.w13_qzeros,
            quant_info.w13_g_idx,
            quant_info.w13_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type1,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        # Hook: after gate_up
        if hooks.after_gate_up:
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)
            hooks.after_gate_up(
                hidden_states, intermediate_cache1_3d, topk_weights, topk_ids
            )
        # Stage 2: Activation
        gate_up = intermediate_cache1.view(-1, 2 * N)
        if runner_config.swiglu_limit is not None:
            # Match fused_marlin_moe exactly: DSV4 clamps only the positive
            # gate bound and clamps both bounds of the up half before SiLU.
            swiglu_limit_func(
                intermediate_cache2,
                gate_up,
                runner_config.swiglu_limit,
            )
        else:
            silu_and_mul(gate_up, intermediate_cache2)
        # Stage 3: Down (Marlin)
        intermediate_cache3 = moe_wna16_marlin_gemm(
            intermediate_cache2,
            intermediate_cache3,
            quant_info.w2_qweight,
            quant_info.w2_bias,
            quant_info.w2_scales,
            quant_info.w2_global_scale,
            quant_info.w2_qzeros,
            quant_info.w2_g_idx,
            quant_info.w2_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type2,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        intermediate_cache3 = intermediate_cache3.view(M, topk, K)

        # Hook: after down
        if hooks.after_down:
            hooks.after_down(
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )
        # Stage 4: Reduction. Never alias hidden_states even under inplace: the sink
        # forward still reads it (stock fused_experts_none_to_marlin does the same).
        output = torch.empty_like(hidden_states)
        if is_mxfp4_marlin:
            # Match fused_marlin_moe: the down GEMM and down-LoRA hook already
            # apply top-k weights (including DSV4's routed scale), then use the
            # same fixed-order vectorized top-k sum. Applying routed_scaling_factor
            # again here was both numerically wrong and byte-divergent at zero LoRA.
            from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum

            moe_topk_sum(intermediate_cache3, output)
            return StandardCombineInput(hidden_states=output)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        # NOTE: fusion opportunity here
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor)

        return StandardCombineInput(hidden_states=output)
