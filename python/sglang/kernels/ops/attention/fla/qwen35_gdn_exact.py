"""Resolve the exact Qwen3.5-family serving implementation as one unit.

The only public switch is ``--rl-on-policy-target xorl``. ``ServerArgs``
combines it with architecture detection, validates the supported envelope,
and calls the private apply function below before constructing the attention
backend. There are no environment inputs, per-feature switches, diagnostic
counters, reset hooks, or public query API.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

QWEN35_DENSE_ARCHS = frozenset(
    {
        "Qwen3_5ForCausalLM",
        "Qwen3_5TextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    }
)
QWEN35_MOE_ARCHS = frozenset(
    {
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeTextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
QWEN35_EXACT_ARCHS = QWEN35_DENSE_ARCHS | QWEN35_MOE_ARCHS
QWEN35_REQUIRED_BI_OPS = (
    "addmm",
    "bmm",
    "log_softmax",
    "mean",
    "mm",
    "rms_norm",
)

# This tuple does not use the GLM numerical-family resolver. Its sensitive
# sites are selected directly: Gemma-style family-1 RMSNorm through the
# registered BI op, the v1 head/LSE functions, the BI router, the explicit
# ordered combine, and the GDN kernels below. Do not add a second family knob.

_applied = False


def _apply_qwen35_gdn_exact(server_args) -> None:
    """Install the one certified implementation tuple once per worker."""
    global _applied
    if _applied:
        return

    is_moe = bool(server_args.qwen35_gdn_exact_is_moe)
    exact_graph = is_moe and not server_args.disable_cuda_graph

    from sglang.srt.batch_invariant_ops import batch_invariant_ops as _bi_ops
    from sglang.srt.batch_invariant_ops import bi_gemm_configs as _gemm_configs
    from sglang.srt.batch_invariant_ops import bi_gemm_tiera as _tiera
    from sglang.srt.distributed import communication_op as _comm
    from sglang.kernels.ops.attention.fla import bi_gdn_decode as _decode
    from sglang.kernels.ops.attention.fla import bi_gdn_decode_fast as _fast
    from sglang.kernels.ops.attention.fla import bi_gdn_decode_incr as _incr
    from sglang.kernels.ops.attention.fla import bi_gdn_incr_lazy_heal as _heal
    from sglang.kernels.ops.attention.fla import bi_gdn_prefill as _prefill
    from sglang.kernels.ops.attention.fla import layernorm_gated as _norm_gated

    if not _bi_ops.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "The exact Qwen3.5-family serving path requires JIT DeepGEMM"
        )
    _bi_ops._ENABLE_MM_DEEPGEMM = True
    _bi_ops._ENABLE_MM_FALLBACK_VARIANT = False
    _bi_ops._ENABLE_MM_COMPARISON_TEST = False

    # Both admitted checkpoints use the original exact prefill scan and
    # partial-chunk-rescan decode. The dense 0.8B evidence predates the MoE
    # Wave-3 fast stack, so none of those later mechanisms may leak into the
    # dense tuple merely because both models share an HF architecture family.
    _prefill.BI_GDN_PREFILL_ENABLED = True
    _prefill.BI_GDN_SOLVE_TRIL_DECODE = is_moe
    _decode.BI_GDN_DECODE_ENABLED = True
    _decode.BI_GDN_BS1_STATIC = is_moe
    _decode.BI_GDN_DECODE_GRAPH = exact_graph
    _fast.BI_GDN_DECODE_FAST_ENABLED = is_moe
    _fast.BI_GDN_FUSE_SMALL_ENABLED = is_moe
    _incr.BI_GDN_DECODE_INCR_ENABLED = exact_graph
    _incr.BI_GDN_INCR_DEFER_ENABLED = exact_graph
    _incr.BI_GDN_VNEW_SLIM_ENABLED = exact_graph
    _heal.BI_GDN_LAZY_HEAL_ENABLED = exact_graph
    _gemm_configs._force_bi_gemm_config_table(is_moe)
    _norm_gated.set_gdn_norm_rows_per_block_pin(4)
    _tiera.set_tiera_enabled(is_moe)
    _bi_ops.set_router_renorm_fused_enabled(is_moe)
    _bi_ops.set_bi_head_fastpath_enabled(is_moe)
    _comm.set_ordered_combine_fused_enabled(is_moe)

    logger.info(
        "Exact Qwen3.5-family zero-K3 serving resolved: BI GDN "
        "prefill/rescan decode%s, rows-per-block pin, contract lm-head + "
        "decode rescore; resolved tuple=%s",
        (
            ", marshal fast path + decode-scheduled solve, incremental-exact "
            "hybrid + writeback deferral + batched heal, Tier-A BI-GEMM "
            "configs, fused router renorm, fused ordered combine, head fastpath"
            if is_moe
            else " (conservative eager tuple; MoE Wave-3 fast paths disabled)"
        ),
        (
            "qwen3.6-moe:tp8/dp8/ep8/pp1,graph10,radix"
            if is_moe
            else "qwen3.5-dense:tp1/dp1/ep1/pp1,eager,no-radix"
        ),
    )
    _applied = True


__all__ = [
    "QWEN35_DENSE_ARCHS",
    "QWEN35_EXACT_ARCHS",
    "QWEN35_MOE_ARCHS",
    "QWEN35_REQUIRED_BI_OPS",
]
