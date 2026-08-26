# Twin package for sglang.srt.batch_invariant_ops.
#
# The no-op __apply_patch__ is required. Without it the finder falls back to
# copying this package's public attributes onto the upstream package, and once
# a twin submodule has been imported it IS a public attribute here -- so the
# upstream package's submodule attribute would be shadowed by the twin.
# (`__patch_include__ = []` does not work: an empty include set is falsy and
# apply_patch falls through to the name copy.)
#
# The package-level re-export widening (main adds the XoRL names to
# `sglang.srt.batch_invariant_ops`'s __init__) is done here explicitly.


def __apply_patch__(public_mod):
    from sglang.xorl.bi import bi_families_v2 as _v2
    from sglang.xorl.bi import ops_ext as _ext

    for name in (
        "exact_temperature_scale_bf16_logits",
        "exact_temperature_scale_fp32_logits",
        "families_v2_enabled",
        "head_v2_full_logits_with_lse",
        "head_v2_selected_logprob",
        "head_v2_selected_logprob_from_logits",
        "rms_norm_v2",
    ):
        setattr(public_mod, name, getattr(_v2, name))
    for name in (
        "RMS_NORM_FAMILIES",
        "RMS_NORM_FAMILY_NO_RESIDUAL",
        "RMS_NORM_FAMILY_RESIDUAL_TREE",
        "RMSNormFamily",
        "bi_fused_add_rms_norm",
        "bi_lm_head_full_logits",
        "bi_lm_head_selected_logprob",
        "bi_lm_head_selected_logprob_from_logits",
        "bi_rms_norm",
        "bi_router_gemm",
        "bi_router_topk_weights",
        "fused_add_rms_norm_batch_invariant",
        "rms_norm_residual_tree_batch_invariant",
        "is_bi_head_fastpath_enabled",
        "set_bi_head_fastpath_enabled",
        "set_router_renorm_fused_enabled",
    ):
        setattr(public_mod, name, getattr(_ext, name))
    # The submodule twin re-exports the op-gated mode helpers; make them
    # visible at package level too (main's __init__ exports them).
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as _ops

    for name in (
        "get_batch_invariant_ops",
        "is_batch_invariant_op_enabled",
    ):
        setattr(public_mod, name, getattr(_ops, name))
