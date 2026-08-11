from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    MoeRunnerConfig,
    PermuteMethodPool,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore
from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsRunnerCore
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.lora.lora_moe_runners import LoRAHooks

logger = logging.getLogger(__name__)


class MoeRunner:
    def __init__(
        self,
        runner_backend: MoeRunnerBackend,
        config: MoeRunnerConfig,
        lora_enabled: bool = False,
    ):
        self.runner_backend = runner_backend
        self.config = config
        self.lora_enabled = lora_enabled

        # --moe-runner-backend hpc_ops makes the standard dispatcher keep
        # global expert ids (skip_local_expert_mapping), so every MoE layer
        # must actually run the hpc_ops runner. A quant method that falls
        # back to another runner here (e.g. an unquantized MoE never enters
        # the FP8 path) would consume global ids as local ones and misroute
        # tokens under EP>1, so fail loudly at startup instead.
        if get_moe_runner_backend().is_hpc_ops() and not runner_backend.is_hpc_ops():
            raise ValueError(
                "--moe-runner-backend hpc_ops was requested, but this MoE "
                f"layer's quantization method selected the "
                f"'{runner_backend.value}' runner (hpc_ops only supports FP8 "
                "blockwise / per-tensor quantized MoE). Remove "
                "--moe-runner-backend hpc_ops for this model."
            )

        self.fused_func = None

        if runner_backend.is_triton():
            self.runner_core = TritonRunnerCore(config)
        elif runner_backend.is_ascend():
            from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

            self.runner_core = AscendRunnerCore(config)
        elif runner_backend.is_triton_kernels():
            self.runner_core = TritonKernelsRunnerCore(config)
        elif runner_backend.is_deep_gemm():
            self.runner_core = DeepGemmRunnerCore(config)
        elif runner_backend.is_humming():
            from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore

            self.runner_core = HummingRunnerCore(config)
        elif runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerCore

            self.runner_core = AiterRunnerCore(config)
        elif runner_backend.is_marlin():
            if lora_enabled:
                from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore

                self.runner_core = MarlinLoraRunnerCore(config)
            else:
                self.runner_core = None  # Marlin only supports fused path
        elif (
            runner_backend.is_flashinfer_trtllm()
            or runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner_core = None  # FlashInfer TRT-LLM only supports fused path
        elif runner_backend.is_flashinfer_cutedsl():
            self.runner_core = None  # FlashInfer CuteDSL only supports fused path
        elif runner_backend.is_flashinfer_cutlass():
            self.runner_core = None  # FlashInfer CUTLASS only supports fused path
        elif runner_backend.is_flashinfer_mxfp4():
            self.runner_core = None  # FlashInfer MXFP4 only supports fused path
            # Import flashinfer_cutlass here (not at module top, to avoid a circular
            # import) to register the flashinfer_mxfp4 fused func before the pool lookup.
            from sglang.srt.layers.moe.moe_runner import (  # noqa: F401
                flashinfer_cutlass,
            )
        elif runner_backend.is_cutlass():
            self.runner_core = None  # CUTLASS uses the direct cutlass_moe_fp4 path
        elif runner_backend.is_hpc_ops():
            import torch

            major, minor = torch.cuda.get_device_capability()
            if major != 9:
                raise ValueError(
                    "--moe-runner-backend hpc_ops requires an SM90 (Hopper) "
                    "GPU (the HPC-Ops kernels ship sm90a only), got "
                    f"sm{major}{minor}."
                )
            self.runner_core = None  # HPC-Ops only supports the fused path
            # Import here (not at module top, to avoid a circular import) to
            # register the hpc_ops fused func before the pool lookup.
            from sglang.srt.layers.moe.moe_runner import hpc_ops  # noqa: F401
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")

        # Skip fused func if LoRA is enabled (LoRA requires non-fused path)
        if not lora_enabled:
            a2a_backend_name = get_moe_a2a_backend().value
            runner_backend_name = runner_backend.value

            # TODO(cwan): add a server argument to disable fused func
            self.fused_func = FusedOpPool.get_fused_func(
                a2a_backend_name, runner_backend_name
            )

            if self.runner_core is None and self.fused_func is None:
                raise NotImplementedError(
                    f"Runner backend {runner_backend} requires a fused func for a2a backend "
                    f"{a2a_backend_name}, but none is registered."
                )

        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        SGLANG_CI_DISABLE_MOE_FUSED_FUNC = os.environ.get(
            "SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "0"
        )
        if SGLANG_CI_DISABLE_MOE_FUSED_FUNC == "1":
            logger.info(
                "SGLANG_CI_DISABLE_MOE_FUSED_FUNC is set to 1, disabling fused func"
            )
            self.fused_func = None

    def run(
        self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo, lora_info=None
    ) -> CombineInput:
        chunked = self._maybe_run_dsv4_exact_marlin_chunks(
            dispatch_output, quant_info, lora_info
        )
        if chunked is not None:
            return chunked

        if self.fused_func is not None and not self.lora_enabled:
            return self.fused_func(dispatch_output, quant_info, self.config)

        assert self.runner_core is not None

        def _maybe_build_lora_hooks(_runner_input: Any) -> LoRAHooks:
            from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutput
            from sglang.srt.lora.lora_moe_runners import build_lora_hooks

            if isinstance(_runner_input, DispatchOutput):
                hidden_states, topk_ids = (
                    _runner_input.hidden_states,
                    _runner_input.topk_output.topk_ids,
                )
            else:
                hidden_states = _runner_input.hidden_states
                topk_ids = getattr(_runner_input, "topk_ids", None)
            if self.lora_enabled and lora_info is not None:
                return build_lora_hooks(
                    hidden_states,
                    lora_info,
                    topk_ids,
                )
            return None

        # Runners that handle dispatch_output directly (e.g., MarlinRunnerCore)
        # bypass the pre-permute step and do their own alignment internally.
        if hasattr(self.runner_core, "run_from_dispatch"):
            hooks = _maybe_build_lora_hooks(dispatch_output)
            return self.runner_core.run_from_dispatch(
                dispatch_output, quant_info, self.config, hooks=hooks
            )

        dispatch_format = dispatch_output.format.value
        runner_format = self.runner_core.runner_backend.value
        self.pre_permute_func = PermuteMethodPool.get_pre_permute(
            dispatch_format, runner_format
        )

        running_state = {}
        if self.down_gemm_overlap_args is not None:
            running_state["down_gemm_overlap_args"] = self.down_gemm_overlap_args
        if self.meta_overlap_args is not None:
            running_state["meta_overlap_args"] = self.meta_overlap_args

        runner_input = self.pre_permute_func(
            dispatch_output, quant_info, self.config, running_state
        )

        hooks = _maybe_build_lora_hooks(runner_input)

        runner_output = self.runner_core.run(
            runner_input, quant_info, running_state, hooks=hooks
        )
        runner_format = self.runner_core.runner_backend.value
        combine_format = dispatch_output.format.value
        self.post_permute_func = PermuteMethodPool.get_post_permute(
            runner_format, combine_format
        )
        combine_input = self.post_permute_func(
            runner_output, quant_info, self.config, running_state
        )

        return combine_input

    def _maybe_run_dsv4_exact_marlin_chunks(
        self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo, lora_info
    ) -> Optional[CombineInput]:
        """DSV4 exact byte program: chunk Marlin token batches.

        A hot expert with more routed rows than one 64-row Marlin block spans
        several blocks, whose cross-block reduction order is not byte-stable
        (see ``is_dsv4_exact_pinned_marlin_geometry``). Chunking so every
        expert stays within one block per launch is part of the DSV4 exact
        byte contract; the trainer chunks identically inside
        ``fused_marlin_moe``. Returns ``None`` for every other model or
        geometry, leaving the stock path untouched.
        """
        import torch

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            DSV4_EXACT_MARLIN_MAX_CHUNK_TOKENS,
            is_dsv4_exact_pinned_marlin_geometry,
        )
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if not isinstance(quant_info, MarlinMoeQuantInfo):
            return None
        if not isinstance(dispatch_output, StandardDispatchOutput):
            return None
        hidden_states = dispatch_output.hidden_states
        num_tokens = hidden_states.shape[0]
        if num_tokens <= DSV4_EXACT_MARLIN_MAX_CHUNK_TOKENS:
            return None
        if get_is_capture_mode():
            return None
        topk_output = dispatch_output.topk_output
        local_experts = quant_info.w13_qweight.shape[0]
        hidden_size = hidden_states.shape[1]
        intermediate_size = quant_info.w2_qweight.shape[1] * 16
        is_mxfp4_marlin = (
            quant_info.weight_bits == 4
            and quant_info.w13_qzeros is None
            and quant_info.w2_qzeros is None
            and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
            and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        )
        global_experts = (
            quant_info.global_num_experts
            if quant_info.global_num_experts != -1
            else local_experts
        )
        if not is_dsv4_exact_pinned_marlin_geometry(
            is_mxfp4_marlin=is_mxfp4_marlin,
            global_experts=global_experts,
            local_experts=local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topk=topk_output.topk_ids.shape[1],
            clamp_limit=self.config.swiglu_limit,
        ):
            return None

        from sglang.srt.lora.lora_moe_runners import slice_moe_lora_info

        chunk = DSV4_EXACT_MARLIN_MAX_CHUNK_TOKENS
        outputs = []
        for start in range(0, num_tokens, chunk):
            stop = min(start + chunk, num_tokens)
            sliced_dispatch = dispatch_output._replace(
                hidden_states=hidden_states[start:stop],
                hidden_states_scale=(
                    dispatch_output.hidden_states_scale[start:stop]
                    if dispatch_output.hidden_states_scale is not None
                    else None
                ),
                topk_output=topk_output._replace(
                    topk_weights=topk_output.topk_weights[start:stop],
                    topk_ids=topk_output.topk_ids[start:stop],
                    router_logits=topk_output.router_logits[start:stop],
                ),
            )
            outputs.append(
                self.run(
                    sliced_dispatch,
                    quant_info,
                    lora_info=slice_moe_lora_info(lora_info, start, stop),
                ).hidden_states
            )
        return StandardCombineInput(hidden_states=torch.cat(outputs, dim=0))

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        self.down_gemm_overlap_args = down_gemm_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.down_gemm_overlap_args = None
        self.meta_overlap_args = None
