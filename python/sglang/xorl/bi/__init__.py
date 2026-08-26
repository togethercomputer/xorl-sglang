# XoRL batch-invariant exact-serving ops, ported from xorl-sglang `main`
# (there they live inside sglang/srt/batch_invariant_ops/). This package
# carries the additive modules; upstream-behaviour changes (op-gated
# enable/disable) live in the sglang.overrides twin for
# sglang.srt.batch_invariant_ops.batch_invariant_ops, which also re-exports
# these names onto the public package so main-lineage import sites work.

from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    AttentionBlockSize,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_batch_invariant_attention_block_size,
    is_batch_invariant_mode_enabled,
    log_softmax,
    matmul_persistent,
    mean_dim,
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
)
from sglang.xorl.bi.bi_families_v2 import (
    exact_temperature_scale_bf16_logits,
    exact_temperature_scale_fp32_logits,
    families_v2_enabled,
    head_v2_full_logits_with_lse,
    head_v2_selected_logprob,
    head_v2_selected_logprob_from_logits,
    rms_norm_v2,
)
from sglang.xorl.bi.ops_ext import (
    RMS_NORM_FAMILIES,
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNormFamily,
    bi_fused_add_rms_norm,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob,
    bi_lm_head_selected_logprob_from_logits,
    bi_rms_norm,
    bi_router_gemm,
    bi_router_topk_weights,
    fused_add_rms_norm_batch_invariant,
    is_bi_head_fastpath_enabled,
    rms_norm_residual_tree_batch_invariant,
    set_bi_head_fastpath_enabled,
    set_router_renorm_fused_enabled,
)

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
    "rms_norm_batch_invariant",
    "bi_lm_head_full_logits",
    "bi_lm_head_selected_logprob",
    "bi_lm_head_selected_logprob_from_logits",
    "bi_router_gemm",
    "bi_router_topk_weights",
    "exact_temperature_scale_bf16_logits",
    "exact_temperature_scale_fp32_logits",
    "families_v2_enabled",
    "head_v2_full_logits_with_lse",
    "head_v2_selected_logprob",
    "head_v2_selected_logprob_from_logits",
    "rms_norm_v2",
    "fused_add_rms_norm_batch_invariant",
    "rms_norm_residual_tree_batch_invariant",
    "RMSNormFamily",
    "RMS_NORM_FAMILY_NO_RESIDUAL",
    "RMS_NORM_FAMILY_RESIDUAL_TREE",
    "RMS_NORM_FAMILIES",
    "bi_rms_norm",
    "bi_fused_add_rms_norm",
    "is_bi_head_fastpath_enabled",
    "set_bi_head_fastpath_enabled",
    "set_router_renorm_fused_enabled",
]
