from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_mxfp8: bool = False
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        if self.config.deepep_native_exact and runner_input.hidden_states.shape[0] == 0:
            return TritonRunnerOutput(hidden_states=runner_input.hidden_states.clone())

        deterministic_ll_routes = bool(
            self.config.deepep_native_exact
            and running_state.get("deepep_ll_deterministic_routes", False)
        )
        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine or deterministic_ll_routes,
            inplace=False if deterministic_ll_routes else self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            # Exact routes keep one base destination program whether an
            # active adapter is present or its LoRA-B is identically zero.
            force_intermediate_output=self.config.deepep_native_exact,
            swiglu_limit=self.config.swiglu_limit,
            # Match normal no_combine=False exactly. The down epilogue first
            # stores BF16(base * route_weight), then the active-LoRA hook adds
            # its weighted delta into that same BF16 route buffer. Keeping the
            # multiply here preserves the normal-mode rounding boundary; the
            # LL combine therefore consumes these BF16 routes with unit weights.
            preweight_no_combine_routes=deterministic_ll_routes,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        # SGLANG_OPT_MOE_QUANT_ONCE: use the caller's pre-quantized activation
        # (per-token-group-128 fp8 q + scales) instead of re-quantizing inside
        # invoke_fused_moe_kernel.
        pre_quant = dispatch_output.hidden_states_pre_quant
        if pre_quant is not None:
            a1_q, a1_scale = pre_quant
        else:
            a1_q, a1_scale = None, quant_info.a13_scale

        output = fused_experts(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=runner_config,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            a1_q=a1_q,
        )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


@register_pre_permute("deepep_normal", "triton")
def pre_permute_deepep_normal_to_triton(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Present real DeepEP receive rows to the ordinary BF16 Triton runner."""

    if dispatch_output.hidden_states_scale is not None:
        raise ValueError("deepep_native_exact rejects quantized DeepEP activations")
    if dispatch_output.hidden_states.dtype is not torch.bfloat16:
        raise ValueError(
            f"deepep_native_exact requires BF16 receive values, got {dispatch_output.hidden_states.dtype}"
        )
    from sglang.srt.layers.moe.deepep_native_exact import (  # noqa: PLC0415
        adapt_native_runner_metadata,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    runner_topk_ids, runner_topk_weights = adapt_native_runner_metadata(
        dispatch_output.topk_ids,
        dispatch_output.topk_weights,
    )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        dispatch_output.hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        runner_topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )
    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    running_state["deepep_native_topk_ids"] = runner_topk_ids
    running_state["deepep_native_topk_weights"] = runner_topk_weights
    return TritonRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        topk_weights=runner_topk_weights,
        topk_ids=runner_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_deepep_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """Keep receive metadata beside the BF16 local rank leaf for combine."""

    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    if runner_config.no_combine:
        raise ValueError(
            "deepep_native_exact requires the fused no_combine=False local leaf program"
        )
    # Metadata is supplied by the runner input; the generic post-permute API
    # does not pass it directly. Preserve it in running_state at pre-permute.
    local_leaf = runner_output.hidden_states
    if local_leaf.ndim != 2 or local_leaf.dtype is not torch.bfloat16:
        raise ValueError(
            "deepep_native_exact fused runner must return a BF16 [recv_rows, hidden] local leaf"
        )
    if local_leaf.shape[0] != running_state["deepep_native_topk_ids"].shape[0]:
        raise ValueError(
            "deepep_native_exact fused runner leaf does not cover every receive row"
        )
    return DeepEPNormalCombineInput(
        hidden_states=local_leaf.contiguous(),
        topk_ids=running_state["deepep_native_topk_ids"],
        topk_weights=running_state["deepep_native_topk_weights"],
    )


@register_pre_permute("deepep_ll", "triton")
def pre_permute_deepep_ll_to_triton(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Expose valid expert-major LL rows as top-k-one Triton routes."""

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLExactDispatchOutput,
    )

    if not isinstance(dispatch_output, DeepEPLLExactDispatchOutput):
        raise ValueError(
            "deepep_ll -> Triton is reserved for deterministic native-exact dispatch"
        )
    hidden = dispatch_output.hidden_states
    packed_weights = dispatch_output.packed_route_weights
    counts = dispatch_output.masked_m
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError(
            "deterministic low-latency DeepEP rejects quantized activations"
        )
    if hidden.ndim != 3 or hidden.dtype is not torch.bfloat16:
        raise ValueError(
            "deterministic low-latency DeepEP requires BF16 [experts, capacity, hidden]"
        )
    num_local_experts, capacity, hidden_size = hidden.shape
    if packed_weights.shape != (num_local_experts, capacity):
        raise ValueError("packed low-latency route weights do not match receive rows")
    if packed_weights.dtype is not torch.float32 or not packed_weights.is_contiguous():
        raise ValueError("packed low-latency route weights must be contiguous FP32")
    if counts.shape != (num_local_experts,) or counts.dtype is not torch.int32:
        raise ValueError("packed low-latency expert counts must be int32 [experts]")
    if int(quant_info.w13_weight.shape[0]) != num_local_experts:
        raise ValueError(
            "low-latency receive experts do not match local expert weights"
        )

    device = hidden.device
    capacity_rows = torch.arange(capacity, device=device, dtype=torch.int32)
    expert_rows = torch.arange(
        num_local_experts, device=device, dtype=torch.int32
    ).unsqueeze(1)
    valid = capacity_rows.unsqueeze(0) < counts.unsqueeze(1)
    runner_topk_ids = torch.where(
        valid, expert_rows, torch.full((), -1, dtype=torch.int32, device=device)
    ).reshape(-1, 1)
    runner_topk_weights = torch.where(
        valid, packed_weights, torch.zeros((), dtype=torch.float32, device=device)
    ).reshape(-1, 1)
    flat_hidden = hidden.reshape(-1, hidden_size)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        flat_hidden,
        quant_info.w13_weight,
        quant_info.w2_weight,
        runner_topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )
    running_state.update(
        config=config,
        down_config=down_config,
        down_moe_use_tma=down_moe_use_tma,
        deepep_ll_deterministic_routes=True,
        deepep_ll_shape=(num_local_experts, capacity, hidden_size),
        deepep_ll_runner_topk_ids=runner_topk_ids,
        deepep_ll_runner_topk_weights=runner_topk_weights,
        deepep_ll_topk_ids=dispatch_output.topk_ids,
        deepep_ll_topk_weights=dispatch_output.topk_weights,
    )
    return TritonRunnerInput(
        hidden_states=flat_hidden,
        topk_weights=runner_topk_weights,
        topk_ids=runner_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_ll")
def post_permute_triton_to_deepep_ll(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    routes = runner_output.hidden_states
    num_local_experts, capacity, hidden_size = running_state["deepep_ll_shape"]
    expected_shape = (num_local_experts * capacity, 1, hidden_size)
    if routes.shape != expected_shape or routes.dtype is not torch.bfloat16:
        raise ValueError(
            f"deterministic low-latency runner returned {tuple(routes.shape)} {routes.dtype}; "
            f"expected BF16 {expected_shape}"
        )
    from sglang.srt.layers.moe.deepep_native_exact import (  # noqa: PLC0415
        pack_native_low_latency_bf16_routes,
    )

    # The runner output is the same weighted BF16 route buffer consumed by the
    # normal-mode local top-k reducer: BF16(base * weight), followed by the
    # weighted active-LoRA down delta. Communicate those BF16 bytes unchanged
    # and use unit combine weights so DeepEP forms the same FP32 owner-rank sum.
    expert_routes = pack_native_low_latency_bf16_routes(
        routes,
        running_state["deepep_ll_runner_topk_ids"],
    )
    return DeepEPLLCombineInput(
        hidden_states=expert_routes.reshape(num_local_experts, capacity, hidden_size),
        topk_ids=running_state["deepep_ll_topk_ids"],
        topk_weights=torch.ones_like(running_state["deepep_ll_topk_weights"]),
    )
