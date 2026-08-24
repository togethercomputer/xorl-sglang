"""Marlin MoE runner core with hook support for LoRA injection.

Uses Marlin int4/int8 kernels for the base MoE projections.
LoRA deltas are injected via hooks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
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
_dsv4_marlin_stage_capture_done = False

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


def _validate_low_latency_counts(counts: torch.Tensor, capacity: int) -> None:
    """Validate DeepEP receive counts without synchronizing a captured stream.

    ``bool(torch.any(...))`` copies a device predicate to the host.  That is a
    useful fail-closed check in eager execution, but CUDA forbids the copy
    while a full decode graph is being captured.  The low-latency DeepEP ABI
    already bounds ``masked_m`` by the configured capacity; during capture and
    replay the device-side mask below remains the source of truth.
    """

    if counts.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    if bool(torch.any(counts < 0)) or bool(torch.any(counts > capacity)):
        raise ValueError("packed low-latency counts are outside receive capacity")


def _prepare_low_latency_marlin_dispatch(
    dispatch_output,
    quant_info: MarlinMoeQuantInfo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """Flatten deterministic expert-major receives into top-1 Marlin rows."""

    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLExactDispatchOutput,
    )

    if not isinstance(dispatch_output, DeepEPLLExactDispatchOutput):
        raise ValueError(
            "MXFP4-Marlin low-latency execution requires deterministic "
            "DeepEP dispatch with packed route weights"
        )
    hidden = dispatch_output.hidden_states
    packed_weights = dispatch_output.packed_route_weights
    counts = dispatch_output.masked_m
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError("deterministic MXFP4-Marlin rejects quantized activations")
    if hidden.ndim != 3 or hidden.dtype is not torch.bfloat16:
        raise ValueError(
            "deterministic MXFP4-Marlin requires BF16 "
            "[local_experts, capacity, hidden] receives"
        )
    num_local_experts, capacity, hidden_size = hidden.shape
    if packed_weights.shape != (num_local_experts, capacity):
        raise ValueError("packed low-latency weights do not match receive rows")
    if packed_weights.dtype is not torch.float32 or not packed_weights.is_contiguous():
        raise ValueError("packed low-latency weights must be contiguous FP32")
    if counts.shape != (num_local_experts,) or counts.dtype is not torch.int32:
        raise ValueError("packed low-latency counts must be int32 [local_experts]")
    _validate_low_latency_counts(counts, capacity)
    if int(quant_info.w13_qweight.shape[0]) != num_local_experts:
        raise ValueError("low-latency receive experts do not match Marlin weights")

    capacity_rows = torch.arange(capacity, device=hidden.device, dtype=torch.int32)
    expert_rows = torch.arange(
        num_local_experts, device=hidden.device, dtype=torch.int32
    ).unsqueeze(1)
    valid = capacity_rows.unsqueeze(0) < counts.unsqueeze(1)
    runner_topk_ids = torch.where(
        valid,
        expert_rows,
        torch.full((), -1, dtype=torch.int32, device=hidden.device),
    ).reshape(-1, 1)
    runner_topk_weights = torch.where(
        valid,
        packed_weights,
        torch.zeros((), dtype=torch.float32, device=hidden.device),
    ).reshape(-1, 1)
    return (
        hidden.reshape(-1, hidden_size),
        runner_topk_ids,
        runner_topk_weights,
        (num_local_experts, capacity, hidden_size),
    )


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
    expert_block_partition_count: int,
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
            expert_block_partition_count=expert_block_partition_count,
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
    mul_topk_weights: bool,
    expert_block_partition_count: int,
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
            mul_topk_weights=mul_topk_weights,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type,
            size_m=(stop - start) * topk,
            size_n=hidden_size,
            size_k=intermediate_size,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            expert_block_partition_count=expert_block_partition_count,
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

    def lora_hook_input(
        self,
        dispatch_output,
        quant_info: MarlinMoeQuantInfo,
    ):
        """Expose the same packed-row geometry that low-latency Marlin runs."""

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if not DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            return dispatch_output
        hidden_states, topk_ids, topk_weights, _ = _prepare_low_latency_marlin_dispatch(
            dispatch_output, quant_info
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MarlinMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks=None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        assert hooks is not None, "hooks must be provided for MarlinLoraRunnerCore"

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        is_deepep_normal = DispatchOutputChecker.format_is_deepep_normal(
            dispatch_output
        )
        is_deepep_ll = DispatchOutputChecker.format_is_deepep_ll(dispatch_output)
        low_latency_shape = None
        if is_deepep_normal:
            hidden_states = dispatch_output.hidden_states
            if runner_config.deepep_native_exact:
                from sglang.srt.layers.moe.deepep_native_exact import (
                    canonicalize_native_routing_metadata,
                )

                topk_weights = canonicalize_native_routing_metadata(
                    dispatch_output.topk_weights
                )
                topk_ids = dispatch_output.topk_ids.to(torch.int32).contiguous()
            else:
                topk_weights = dispatch_output.topk_weights
                topk_ids = dispatch_output.topk_ids
        elif is_deepep_ll:
            (
                hidden_states,
                topk_ids,
                topk_weights,
                low_latency_shape,
            ) = _prepare_low_latency_marlin_dispatch(
                dispatch_output,
                quant_info,
            )
        else:
            hidden_states = dispatch_output.hidden_states
            topk_output = dispatch_output.topk_output
            topk_weights = topk_output.topk_weights
            topk_ids = topk_output.topk_ids

        global _dsv4_marlin_stage_capture_done
        stage_capture_dir = os.environ.get(
            "XORL_DSV4_SAMPLER_MARLIN_STAGE_CAPTURE_DIR", ""
        ).strip()
        capture_stages = (
            bool(stage_capture_dir)
            and not _dsv4_marlin_stage_capture_done
            and runner_config.dsv4_exact_mode
            and runner_config.layer_id == 0
            and is_deepep_normal
            and hooks.after_gate_up is not None
            and hooks.after_down is not None
        )
        stage_state = (
            {
                "hidden_states": hidden_states.detach().clone(),
                "topk_ids": topk_ids.detach().clone(),
                "topk_weights": topk_weights.detach().clone(),
            }
            if capture_stages
            else None
        )

        # DeepEP normal dispatch can produce an empty receive batch on a rank.
        # No base or LoRA kernel is needed, but combine must still receive the
        # exact empty routed payload to complete the collective.
        if is_deepep_normal and hidden_states.shape[0] == 0:
            if runner_config.no_combine:
                from sglang.srt.layers.moe.deepep_native_exact import (
                    native_zero_row_runner_routes,
                    reduce_native_runner_routes_to_bf16,
                )

                hidden_states = reduce_native_runner_routes_to_bf16(
                    native_zero_row_runner_routes(hidden_states, topk_ids),
                    topk_ids,
                    topk_weights,
                )
            from sglang.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPNormalCombineInput,
            )

            return DeepEPNormalCombineInput(
                hidden_states=hidden_states.clone(),
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

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

        # Low-latency DeepEP packs expert-capacity rows as top-k 1 for the
        # Marlin launch, but that is a transport layout rather than the model's
        # arithmetic geometry.  DSV4's admitted normal program is selected by
        # its logical top-k 6 and pins block_size_m=64.  Feeding the packed
        # top-k 1 here silently selected the generic block-size-8 program and
        # changed MXFP4 accumulation relative to normal prefill/training.
        logical_topk = (
            int(dispatch_output.topk_ids.shape[1])
            if is_deepep_ll and runner_config.dsv4_exact_mode
            else topk
        )
        block_size_m = select_marlin_moe_block_size_m(
            dsv4_exact_mode=runner_config.dsv4_exact_mode,
            num_tokens=1 if singleton_base else M,
            topk=logical_topk,
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
        expert_block_partition_count = (
            topk
            if runner_config.dsv4_exact_mode and is_mxfp4_marlin and not is_deepep_ll
            else 1
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
            expert_block_partition_count=expert_block_partition_count,
        )
        if stage_state is not None:
            stage_state["gate_up_base"] = intermediate_cache1.detach().clone()
        # Hook: after gate_up
        if hooks.after_gate_up:
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)
            hooks.after_gate_up(
                hidden_states, intermediate_cache1_3d, topk_weights, topk_ids
            )
        if stage_state is not None:
            stage_state["gate_up_after_lora"] = intermediate_cache1.detach().clone()
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
        if stage_state is not None:
            stage_state["activated"] = intermediate_cache2.detach().clone()
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
            mul_topk_weights=(
                not runner_config.no_combine
                and (
                    not is_deepep_ll
                    or runner_config.deepep_native_exact_defer_routed_scale
                )
            ),
            expert_block_partition_count=expert_block_partition_count,
        )
        if stage_state is not None:
            stage_state["down_base"] = intermediate_cache3.detach().clone()
        intermediate_cache3 = intermediate_cache3.view(M, topk, K)

        # Hook: after down
        if hooks.after_down:
            hooks.after_down(
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )
        if stage_state is not None:
            stage_state["down_after_lora"] = intermediate_cache3.detach().clone()
        if is_deepep_ll:
            from sglang.srt.layers.moe.deepep_native_exact import (
                pack_native_low_latency_bf16_routes,
            )
            from sglang.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPLLCombineInput,
            )

            assert low_latency_shape is not None
            num_local_experts, capacity, hidden_size = low_latency_shape
            if runner_config.deepep_native_exact_defer_routed_scale:
                # Match DSV4 normal mode: Marlin and its active-LoRA hook have
                # already applied the route coefficient. The model's 1.5
                # routed scale remains owned by the routed/shared join.
                packed_routes = pack_native_low_latency_bf16_routes(
                    intermediate_cache3,
                    topk_ids,
                )
                combine_topk_weights = torch.ones_like(dispatch_output.topk_weights)
            else:
                # Keep unweighted expert results BF16 on the wire. DeepEP owns
                # the FP32 route multiply, fused-epilogue BF16 route rounding,
                # FP32 owner-rank sum, model scale, and BF16 rank-leaf cast.
                packed_routes = pack_native_low_latency_bf16_routes(
                    intermediate_cache3,
                    topk_ids,
                )
                combine_topk_weights = dispatch_output.topk_weights
            packed_routes = packed_routes.view(num_local_experts, capacity, hidden_size)
            return DeepEPLLCombineInput(
                hidden_states=packed_routes,
                topk_ids=dispatch_output.topk_ids,
                topk_weights=combine_topk_weights,
            )
        if runner_config.no_combine:
            if not DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
                return StandardCombineInput(
                    hidden_states=intermediate_cache3.contiguous()
                )
            from sglang.srt.layers.moe.deepep_native_exact import (
                reduce_native_runner_routes_to_bf16,
            )
            from sglang.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPNormalCombineInput,
            )

            return DeepEPNormalCombineInput(
                hidden_states=reduce_native_runner_routes_to_bf16(
                    intermediate_cache3,
                    topk_ids,
                    topk_weights,
                ),
                topk_ids=topk_ids,
                topk_weights=topk_weights,
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
            if stage_state is not None:
                import torch.distributed as dist

                _dsv4_marlin_stage_capture_done = True
                os.makedirs(stage_capture_dir, exist_ok=True)
                global_rank = dist.get_rank() if dist.is_initialized() else 0
                stage_state["output"] = output.detach().clone()
                torch.save(
                    {
                        "schema": "xorl.dsv4_sampler_marlin_stages.v1",
                        "global_rank": global_rank,
                        "layer_id": runner_config.layer_id,
                        **{name: value.cpu() for name, value in stage_state.items()},
                    },
                    os.path.join(
                        stage_capture_dir,
                        f"rank{global_rank:05d}.layer000.marlin-stages.pt",
                    ),
                )
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
                from sglang.srt.layers.moe.token_dispatcher.deepep import (
                    DeepEPNormalCombineInput,
                )

                return DeepEPNormalCombineInput(
                    hidden_states=output,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
            return StandardCombineInput(hidden_states=output)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        # NOTE: fusion opportunity here
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor)

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            from sglang.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPNormalCombineInput,
            )

            return DeepEPNormalCombineInput(
                hidden_states=output,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        return StandardCombineInput(hidden_states=output)
