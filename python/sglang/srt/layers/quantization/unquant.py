from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
import triton.language as tl
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.lora.moe import build_chunked_compound_segments_cpu
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Skip aiter weight shuffle when using non-auto MoE backend (e.g., triton, triton_kernels)
        # because aiter CK kernels don't support all GEMM dimensions
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        if _should_use_aiter_moe:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )

        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                weight.data = weight.data.transpose(1, 2)
                weight.data = npu_format_cast(
                    weight.data,
                )

        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = MoeRunnerBackend.FLASHINFER_TRTLLM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def _supports_moe_lora(self, layer: torch.nn.Module) -> bool:
        return (
            getattr(layer, "moe_lora_backend", None) is not None
            and getattr(layer, "moe_lora_present", None) is not None
        )

    def _has_active_moe_lora_for_layer(
        self, layer: torch.nn.Module, num_tokens: int
    ) -> bool:
        return self._supports_moe_lora(layer) and layer.has_active_moe_lora(num_tokens)

    def _validate_moe_lora_runtime(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        dispatch_output: StandardDispatchOutput,
    ) -> None:
        batch_info = getattr(layer.moe_lora_backend, "batch_info", None)
        if getattr(layer.moe_lora_backend, "name", None) != "csgmv":
            raise NotImplementedError(
                "MoE LoRA currently requires --lora-backend csgmv."
            )
        if batch_info is None or batch_info.token_weight_indices is None:
            raise NotImplementedError(
                "MoE LoRA requires per-token adapter ids to be materialized in the LoRA batch info."
            )
        if any(
            getattr(layer, attr, None) is None
            for attr in (
                "moe_lora_gate_up_A",
                "moe_lora_gate_B",
                "moe_lora_up_B",
                "moe_lora_down_A",
                "moe_lora_down_B",
            )
        ):
            raise NotImplementedError(
                "MoE LoRA requires gate_up_proj and down_proj LoRA buffers to be initialized."
            )
        if self.use_triton_kernels:
            raise NotImplementedError(
                "MoE LoRA is not implemented for the triton_kernels MoE backend."
            )
        if self.use_flashinfer_cutlass or self.use_flashinfer_trtllm_moe:
            raise NotImplementedError(
                "MoE LoRA is only implemented for the unquantized Triton MoE backend."
            )
        if layer.moe_tp_size > 1 or layer.moe_ep_size > 1:
            raise NotImplementedError(
                "MoE LoRA currently requires TP=1 and EP=1."
            )
        if layer.num_fused_shared_experts > 0:
            raise NotImplementedError(
                "MoE LoRA does not support fused shared experts."
            )
        if self.moe_runner_config.no_combine:
            raise NotImplementedError("MoE LoRA does not support no_combine=True.")
        if self.moe_runner_config.apply_router_weight_on_input:
            raise NotImplementedError(
                "MoE LoRA does not support apply_router_weight_on_input=True."
            )
        if batch_info.token_weight_indices.numel() < x.shape[0]:
            raise NotImplementedError(
                "MoE LoRA batch metadata does not cover the current token count."
            )
        if dispatch_output.topk_output.topk_ids.numel() == 0:
            raise NotImplementedError("MoE LoRA does not support empty routed batches.")
        if bool(torch.any(dispatch_output.topk_output.topk_ids < 0).item()):
            raise NotImplementedError(
                "MoE LoRA currently does not support filtered expert ids."
            )

    def _build_moe_lora_batch_info(
        self,
        layer: torch.nn.Module,
        token_lora_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> LoRABatchInfo:
        num_tokens, topk = topk_ids.shape
        dispatch_lora_ids_cpu = (
            token_lora_ids.to(torch.int32).repeat_interleave(topk).cpu()
        )
        dispatch_expert_ids_cpu = topk_ids.reshape(-1).to(torch.int32).cpu()

        chunk_size = 16
        if hasattr(layer.moe_lora_backend, "_determine_chunk_size_for_tokens"):
            chunk_size = layer.moe_lora_backend._determine_chunk_size_for_tokens(
                num_tokens * topk
            )

        permutation_cpu, seg_weight_indices_cpu, seg_indptr_cpu = (
            build_chunked_compound_segments_cpu(
                dispatch_expert_ids_cpu,
                dispatch_lora_ids_cpu,
                layer.moe_lora_present.shape[0],
                chunk_size,
            )
        )

        return LoRABatchInfo(
            use_cuda_graph=False,
            bs=num_tokens * topk,
            num_segments=len(seg_weight_indices_cpu),
            seg_indptr=seg_indptr_cpu.to(device=topk_ids.device),
            weight_indices=seg_weight_indices_cpu.to(device=topk_ids.device),
            lora_ranks=layer.moe_lora_backend.batch_info.lora_ranks.repeat(
                layer.num_local_experts
            ).contiguous(),
            scalings=layer.moe_lora_backend.batch_info.scalings.repeat(
                layer.num_local_experts
            ).contiguous(),
            max_len=chunk_size,
            seg_lens=None,
            permutation=permutation_cpu.to(device=topk_ids.device),
        )

    def _forward_cuda_moe_lora(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
            get_config_dtype_str,
            invoke_fused_moe_kernel,
            moe_align_block_size,
            moe_sum_reduce,
            moe_sum_reduce_torch_compile,
            moe_sum_reduce_triton,
            silu_and_mul,
            try_get_optimal_moe_config,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.lora.triton_ops import (
            chunked_sgmv_lora_expand_forward,
            chunked_sgmv_lora_shrink_forward,
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        num_tokens, topk = topk_ids.shape
        token_lora_ids = layer.moe_lora_backend.batch_info.token_weight_indices[
            :num_tokens
        ].to(torch.long)
        expert_lora_batch_info = self._build_moe_lora_batch_info(
            layer, token_lora_ids, topk_ids
        )

        config_dtype = get_config_dtype_str(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            dtype=x.dtype,
        )
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            layer.w13_weight.shape,
            (
                layer.w2_weight.shape[0],
                layer.w2_weight.shape[1],
                layer.w2_weight.shape[2],
            ),
            topk,
            config_dtype,
            block_shape=None,
            per_channel_quant=False,
            return_down_config=True,
        )
        config, (down_config, _) = get_config_func(num_tokens)
        compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], layer.num_local_experts
        )

        gate_up_out_dim = layer.w13_weight.shape[1]
        down_out_dim = layer.w2_weight.shape[1]
        base_gate_up = torch.empty(
            (num_tokens * topk, gate_up_out_dim),
            device=x.device,
            dtype=x.dtype,
        )
        invoke_fused_moe_kernel(
            x,
            layer.w13_weight,
            getattr(layer, "w13_weight_bias", None),
            base_gate_up,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            filter_expert=False,
        )

        shared_batch_info = layer.moe_lora_backend.batch_info
        shared_gate_up = chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=layer.moe_lora_gate_up_A,
            batch_info=shared_batch_info,
            num_slices=2,
        )
        dispatch_token_ids = (
            torch.arange(num_tokens, device=x.device, dtype=torch.long)
            .repeat_interleave(topk)
            .contiguous()
        )
        z_disp = shared_gate_up.index_select(0, dispatch_token_ids)

        gate_b = layer.moe_lora_gate_B.view(
            -1,
            layer.moe_lora_gate_B.shape[-2],
            layer.moe_lora_gate_B.shape[-1],
        )
        up_b = layer.moe_lora_up_B.view(
            -1,
            layer.moe_lora_up_B.shape[-2],
            layer.moe_lora_up_B.shape[-1],
        )
        gate_output_offset = torch.tensor(
            [0, layer.intermediate_size_per_partition],
            dtype=torch.int32,
            device=x.device,
        )
        gate_z = z_disp[:, : gate_b.shape[-1]].contiguous()
        up_z = z_disp[:, gate_b.shape[-1] : gate_b.shape[-1] * 2].contiguous()
        gate_delta = chunked_sgmv_lora_expand_forward(
            x=gate_z,
            weights=gate_b,
            batch_info=expert_lora_batch_info,
            slice_offsets=gate_output_offset,
            max_slice_size=layer.intermediate_size_per_partition,
            base_output=None,
        )
        up_delta = chunked_sgmv_lora_expand_forward(
            x=up_z,
            weights=up_b,
            batch_info=expert_lora_batch_info,
            slice_offsets=gate_output_offset,
            max_slice_size=layer.intermediate_size_per_partition,
            base_output=None,
        )
        base_gate_up[:, : layer.intermediate_size_per_partition].add_(gate_delta)
        base_gate_up[:, layer.intermediate_size_per_partition :].add_(up_delta)

        down_input = torch.empty(
            (num_tokens * topk, gate_up_out_dim // 2),
            device=x.device,
            dtype=x.dtype,
        )
        if self.moe_runner_config.activation == "silu":
            if self.moe_runner_config.gemm1_alpha is not None:
                assert self.moe_runner_config.gemm1_clamp_limit is not None
                down_input = _swiglu_gpt_oss_sigmoid_alpha(
                    base_gate_up,
                    self.moe_runner_config.gemm1_alpha,
                    self.moe_runner_config.gemm1_clamp_limit,
                )
            elif self.moe_runner_config.gemm1_clamp_limit is not None:
                down_input = _swiglu_silu_clamp_mul(
                    base_gate_up,
                    self.moe_runner_config.gemm1_clamp_limit,
                )
            else:
                silu_and_mul(base_gate_up, down_input)
        else:
            raise NotImplementedError(
                f"MoE LoRA currently only supports silu activation, got {self.moe_runner_config.activation}."
            )

        base_down = torch.empty(
            (num_tokens, topk, down_out_dim),
            device=x.device,
            dtype=x.dtype,
        )
        invoke_fused_moe_kernel(
            down_input,
            layer.w2_weight,
            getattr(layer, "w2_weight_bias", None),
            base_down,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            filter_expert=False,
        )

        base_out = torch.empty((num_tokens, down_out_dim), device=x.device, dtype=x.dtype)
        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        if topk == 1 and routed_scaling_factor == 1.0:
            base_out.copy_(base_down[:, 0])
        elif topk == 2 and routed_scaling_factor == 1.0:
            torch.add(base_down[:, 0], base_down[:, 1], out=base_out)
        elif num_tokens <= 32:
            moe_sum_reduce_torch_compile(base_down, base_out, routed_scaling_factor)
        else:
            try:
                moe_sum_reduce(base_down, base_out, routed_scaling_factor)
            except NameError:
                moe_sum_reduce_triton(base_down, base_out, routed_scaling_factor)

        down_a = layer.moe_lora_down_A.view(
            -1,
            layer.moe_lora_down_A.shape[-2],
            layer.moe_lora_down_A.shape[-1],
        )
        z_down = chunked_sgmv_lora_shrink_forward(
            x=down_input,
            weights=down_a,
            batch_info=expert_lora_batch_info,
            num_slices=1,
        )
        z_reduced = torch.sum(
            z_down.view(num_tokens, topk, -1)
            * topk_weights.to(z_down.dtype).unsqueeze(-1),
            dim=1,
        )

        hidden_output_offset = torch.tensor(
            [0, layer.hidden_size], dtype=torch.int32, device=x.device
        )
        delta = chunked_sgmv_lora_expand_forward(
            x=z_reduced,
            weights=layer.moe_lora_down_B,
            batch_info=shared_batch_info,
            slice_offsets=hidden_output_offset,
            max_slice_size=layer.hidden_size,
            base_output=None,
        )
        if routed_scaling_factor != 1.0:
            delta.mul_(routed_scaling_factor)
        base_out.add_(delta)
        return StandardCombineInput(hidden_states=base_out)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if self._has_active_moe_lora_for_layer(layer, x.shape[0]):
            self._validate_moe_lora_runtime(layer, x, dispatch_output)
            return self._forward_cuda_moe_lora(layer, dispatch_output)

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            output = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_output.topk_ids,
                token_final_scales=topk_output.topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=(
                    ActivationType.Relu2
                    if moe_runner_config.activation == "relu2"
                    else ActivationType.Swiglu
                ),
            )[0]
            return StandardCombineInput(hidden_states=output)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            # Skip aiter fused_moe when using non-auto MoE backend (e.g., triton, triton_kernels)
            # because aiter CK kernels don't support all GEMM dimensions
            _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
            if _should_use_aiter_moe:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                    expert_mask=layer.expert_mask_gpu,
                )
                return StandardCombineInput(hidden_states=output)
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self._has_active_moe_lora_for_layer(layer, dispatch_output.hidden_states.shape[0]):
            raise NotImplementedError(
                "MoE LoRA is only implemented for CUDA decode on the unquantized Triton MoE backend."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self._has_active_moe_lora_for_layer(layer, dispatch_output.hidden_states.shape[0]):
            raise NotImplementedError(
                "MoE LoRA is only implemented for CUDA decode on the unquantized Triton MoE backend."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self._has_active_moe_lora_for_layer(layer, dispatch_output.hidden_states.shape[0]):
            raise NotImplementedError(
                "MoE LoRA is only implemented for CUDA decode on the unquantized Triton MoE backend."
            )

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu
