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

# This tuple does not use the GLM numerical-family resolver. Exact Qwen MoE
# structurally uses the common canonical contributor fold; its head/LSE,
# router, and GDN kernels remain independently resolved.

_applied = False


def _apply_qwen35_gdn_exact(server_args) -> None:
    """Install the one certified implementation tuple once per worker."""
    global _applied
    if _applied:
        return

    is_moe = bool(server_args.qwen35_gdn_exact_is_moe)
    exact_graph = is_moe and not server_args.disable_cuda_graph
    rmsnorm_family = getattr(server_args, "qwen35_rmsnorm_family", "v2")
    if rmsnorm_family not in ("v1", "v2"):
        raise RuntimeError(f"Unsupported exact Qwen RMSNorm family: {rmsnorm_family!r}")

    from sglang.kernels.ops.attention.fla import bi_gdn_decode as _decode
    from sglang.kernels.ops.attention.fla import bi_gdn_decode_fast as _fast
    from sglang.kernels.ops.attention.fla import bi_gdn_decode_incr as _incr
    from sglang.kernels.ops.attention.fla import bi_gdn_incr_lazy_heal as _heal
    from sglang.kernels.ops.attention.fla import bi_gdn_prefill as _prefill
    from sglang.kernels.ops.attention.fla import layernorm_gated as _norm_gated
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as _bi_ops
    from sglang.srt.batch_invariant_ops import bi_gemm_configs as _gemm_configs
    from sglang.srt.batch_invariant_ops import bi_gemm_tiera as _tiera
    from sglang.srt.distributed.canonical_moe import CANONICAL_MOE_FOLD_VERSION

    if not _bi_ops.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "The exact Qwen3.5-family serving path requires JIT DeepGEMM"
        )
    _bi_ops._ENABLE_MM_DEEPGEMM = True
    _bi_ops._ENABLE_MM_FALLBACK_VARIANT = False
    _bi_ops._ENABLE_MM_COMPARISON_TEST = False

    # The production literal-zero receipt was taken with the original exact
    # prefill scan, partial-chunk-rescan graph decode, and conservative
    # no-overlap/no-padding serving schedule.  Live Qwen3.6 GRPO gates measured
    # nonzero K3 for both the promoted Wave-3 tuple and the conservative GDN
    # tuple when the newer overlap + padded-graph schedule remained selected.
    # Keep every unpromoted mechanism out of the architecture-owned tuple until
    # a fresh live gate promotes it; component equality is not a substitute for
    # the trainer-versus-sampler denominator.
    _prefill.BI_GDN_PREFILL_ENABLED = True
    _prefill.BI_GDN_SOLVE_TRIL_DECODE = False
    _decode.BI_GDN_DECODE_ENABLED = True
    _decode.BI_GDN_BS1_STATIC = is_moe
    _decode.BI_GDN_DECODE_GRAPH = exact_graph
    # The modern attention backend uses the slot-direct transport for exact
    # graph replay.  Its arithmetic stages are the oracle prefill binaries;
    # unlike the retired single-bucket staging path it supports every exact
    # decode graph shape through the graph-32 ceiling.
    _fast.BI_GDN_DECODE_FAST_ENABLED = is_moe
    _fast.BI_GDN_FUSE_SMALL_ENABLED = False
    _incr.BI_GDN_DECODE_INCR_ENABLED = False
    _incr.BI_GDN_INCR_DEFER_ENABLED = False
    _incr.BI_GDN_VNEW_SLIM_ENABLED = False
    _heal.BI_GDN_LAZY_HEAL_ENABLED = False
    _gemm_configs._force_bi_gemm_config_table(is_moe)
    _norm_gated.set_gdn_norm_rows_per_block_pin(4)
    _tiera.set_tiera_enabled(False)
    _bi_ops.set_router_renorm_fused_enabled(False)
    _bi_ops.set_bi_head_fastpath_enabled(False)

    logger.info(
        "Exact Qwen3.5-family zero-K3 serving resolved: BI GDN "
        "prefill/rescan decode%s, rows-per-block pin, contract lm-head + "
        "decode rescore; rmsnorm_family=%s; moe_fold=%s; resolved tuple=%s",
        (
            ", conservative no-overlap/no-padding partial-chunk-rescan graph "
            "program; Wave-3 fast mechanisms held behind live zero-K3 promotion"
            if is_moe
            else " (conservative eager tuple; MoE Wave-3 fast paths disabled)"
        ),
        rmsnorm_family,
        CANONICAL_MOE_FOLD_VERSION if is_moe else "none",
        (
            "qwen3.6-moe:tp8/dp8/ep8/pp1,graph32,no-radix,full-prefill"
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
