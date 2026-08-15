"""Marlin MoE runner core with hook support for LoRA injection.

Uses Marlin int4/int8 kernels for the base MoE projections.
LoRA deltas are injected via hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    should_singleton_mxfp4_marlin_base,
)
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


MarlinAlignment = tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]


def _build_marlin_alignments(
    *,
    topk_ids: torch.Tensor,
    block_size_m: int,
    num_experts: int,
    singleton: bool,
) -> list[MarlinAlignment]:
    token_ranges = (
        [(token, token + 1) for token in range(topk_ids.shape[0])]
        if singleton
        else [(0, topk_ids.shape[0])]
    )
    return [
        (
            start,
            stop,
            *moe_align_block_size(
                topk_ids[start:stop],
                block_size_m,
                num_experts,
            ),
        )
        for start, stop in token_ranges
    ]


def _run_marlin_gate_up_base(
    *,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    topk_weights: torch.Tensor,
    alignments: list[MarlinAlignment],
    workspace: torch.Tensor,
    block_size_m: int,
    topk: int,
    intermediate_size: int,
    hidden_size: int,
    scalar_type,
    use_atomic_add: bool,
) -> None:
    for start, stop, sorted_token_ids, expert_ids, num_tokens_post_padded in alignments:
        route_start = start * topk
        route_stop = stop * topk
        moe_wna16_marlin_gemm(
            hidden_states[start:stop],
            output[route_start:route_stop],
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
            topk_weights[start:stop],
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type,
            size_m=stop - start,
            size_n=2 * intermediate_size,
            size_k=hidden_size,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )


def _run_marlin_down_base(
    *,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    topk_weights: torch.Tensor,
    alignments: list[MarlinAlignment],
    workspace: torch.Tensor,
    block_size_m: int,
    topk: int,
    intermediate_size: int,
    hidden_size: int,
    scalar_type,
    use_atomic_add: bool,
) -> None:
    for start, stop, sorted_token_ids, expert_ids, num_tokens_post_padded in alignments:
        route_start = start * topk
        route_stop = stop * topk
        moe_wna16_marlin_gemm(
            hidden_states[route_start:route_stop],
            output[route_start:route_stop],
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
            topk_weights[start:stop],
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type,
            size_m=(stop - start) * topk,
            size_n=hidden_size,
            size_k=intermediate_size,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )


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
        singleton_base = should_singleton_mxfp4_marlin_base(
            dsv4_exact_mode=runner_config.dsv4_exact_mode,
            is_mxfp4_marlin=is_mxfp4_marlin,
            num_tokens=M,
        )

        block_size_m = select_marlin_moe_block_size_m(
            dsv4_exact_mode=runner_config.dsv4_exact_mode,
            num_tokens=1 if singleton_base else M,
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
        # MXFP4 Marlin changes each expert's K-stripe partition when unrelated
        # routes add local expert blocks to the same launch. Keep only the
        # frozen base GEMMs on a one-logical-token program; the LoRA hooks stay
        # batched so their alignment and parameter program execute once.
        alignments = _build_marlin_alignments(
            topk_ids=topk_ids,
            block_size_m=block_size_m,
            num_experts=align_num_experts,
            singleton=singleton_base,
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
        _run_marlin_gate_up_base(
            hidden_states=hidden_states,
            output=intermediate_cache1,
            quant_info=quant_info,
            topk_weights=topk_weights,
            alignments=alignments,
            workspace=workspace,
            block_size_m=block_size_m,
            topk=topk,
            intermediate_size=N,
            hidden_size=K,
            scalar_type=scalar_type1,
            use_atomic_add=use_atomic_add,
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
        _run_marlin_down_base(
            hidden_states=intermediate_cache2,
            output=intermediate_cache3,
            quant_info=quant_info,
            topk_weights=topk_weights,
            alignments=alignments,
            workspace=workspace,
            block_size_m=block_size_m,
            topk=topk,
            intermediate_size=N,
            hidden_size=K,
            scalar_type=scalar_type2,
            use_atomic_add=use_atomic_add,
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
