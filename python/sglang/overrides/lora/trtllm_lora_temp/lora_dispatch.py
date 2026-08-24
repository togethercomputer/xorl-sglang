"""Override twin of ``sglang.srt.lora.trtllm_lora_temp.lora_dispatch``.

Routes **no-adapter** FP8 batches to the standard flashinfer-TRT-LLM FP8
implementation, the way the bf16 path already does, instead of into this fork's
vendored ``fused_experts_fp8_sgl``.

Why: the vendored path calls ``core.trtllm_fp8_block_scale_routed_moe``, which
positionally invokes flashinfer's ``trtllm_fp8_block_scale_moe``. flashinfer
0.6.15 inserts four parameters -- ``gemm1_lora_delta``, ``gemm1_alpha``,
``gemm1_beta``, ``gemm1_clamp_limit`` -- between ``gemm1_weights_scale`` and
``gemm2_weights``, while this fork's vendored ``.cu`` still builds the older
29-argument op. The old call therefore lands ``gemm2_weights_scale`` in the
``gemm1_alpha`` slot and flashinfer rejects it::

    ValueError: gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only
    supported for Fp8QuantizationType.MxFp8 in FP8 block scale MoE.

Adding the four arguments at the call site is *not* the fix: the C++ op then
reports ``Expected 29 but got 34``. The two sides genuinely disagree, and
reconciling them means updating the vendored launcher (C++ work, tracked in
togethercomputer/experimental-moe-lora-kernel, whose README already lists FP8 as
"blocked before the kernel" for the same class of drift on 0.6.17).

The LoRA path itself is unaffected -- it calls this fork's own
``sgl_trtllm_fp8_block_scale_moe_lora`` op via the *raw* module, bypassing
flashinfer's Python wrapper, and measures 16/16 correct on FP8. Only the
no-adapter branch goes through the drifted wrapper, so only that branch is
redirected here.

This mirrors what ``fused_experts_none_to_experimental_sgl_trtllm_bf16_lora``
already does for bf16 (``lora_dispatch.py`` no-LoRA branch ->
``fused_experts_none_to_flashinfer_trtllm_bf16``), making the two dtypes
consistent.
"""


def __apply_patch__(public_mod):
    orig_fp8_lora = public_mod.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora

    def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora(
        dispatch_output,
        quant_info,
        runner_config,
        lora_info,
    ):
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        # Same predicate the upstream function uses to pick its no-LoRA branch;
        # we intercept it before that branch reaches the drifted wrapper. Under
        # cuda-graph capture the original is left alone: capture must keep taking
        # exactly the path it will replay.
        if not get_is_capture_mode() and not lora_info.has_active_lora:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                fused_experts_none_to_flashinfer_trtllm_fp8,
            )

            return fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                quant_info,
                runner_config,
                use_routed_topk=True,
            )

        return orig_fp8_lora(dispatch_output, quant_info, runner_config, lora_info)

    public_mod.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora
    )
