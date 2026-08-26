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


def _defer_incremental_state_writeback(server_args, *, exact_graph: bool) -> bool:
    """Defer recurrent-state writes only when no radix insertion can observe them."""

    return exact_graph and server_args.disable_radix_cache


def _apply_qwen35_gdn_exact(server_args) -> None:
    """Install the one certified implementation tuple once per worker."""
    global _applied
    if _applied:
        return

    is_moe = bool(server_args.qwen35_gdn_exact_is_moe)
    exact_graph = not server_args.disable_cuda_graph
    slot_direct = is_moe or exact_graph
    rmsnorm_family = getattr(server_args, "qwen35_rmsnorm_family", "v2")
    if rmsnorm_family not in ("v1", "v2"):
        raise RuntimeError(f"Unsupported exact Qwen RMSNorm family: {rmsnorm_family!r}")

    from sglang.xorl.fla import bi_gdn_decode as _decode
    from sglang.xorl.fla import bi_gdn_decode_fast as _fast
    from sglang.xorl.fla import bi_gdn_decode_incr as _incr
    from sglang.xorl.fla import bi_gdn_incr_lazy_heal as _heal
    from sglang.xorl.fla import bi_gdn_prefill as _prefill
    from sglang.kernels.ops.attention.fla import layernorm_gated as _norm_gated
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as _bi_ops
    from sglang.xorl.bi import bi_gemm_configs as _gemm_configs
    from sglang.xorl.bi import bi_gemm_tiera as _tiera
    from sglang.xorl.bi import ops_ext as _bi_ops_ext

    # The canonical MoE contributor fold is not ported to this dev-based
    # branch; ServerArgs rejects the Qwen3.5 MoE architectures outright, so
    # is_moe can never be True here.
    if is_moe:
        raise RuntimeError(
            "Exact Qwen3.5 MoE serving is not ported to this branch; the "
            "canonical contributor fold is main-only."
        )

    if not _bi_ops.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "The exact Qwen3.5-family serving path requires JIT DeepGEMM"
        )
    _bi_ops._ENABLE_MM_DEEPGEMM = True
    _bi_ops._ENABLE_MM_FALLBACK_VARIANT = False
    _bi_ops._ENABLE_MM_COMPARISON_TEST = False

    # Dense Qwen retains its directly certified eager tuple.  MoE uses the
    # component-byte-certified cached-row graph program: true graph decode updates
    # only the new row, while eager fallback remains the exact bounded rescan and
    # maintains the same cached intermediates. Batched cache warm replaces the
    # old request-serial warm loop; its slim composition must persist gcum rows
    # as well as the other cached intermediates.
    _prefill.BI_GDN_PREFILL_ENABLED = True
    _prefill.BI_GDN_SOLVE_TRIL_DECODE = slot_direct
    # FAST, INCR, and HEAL import this scalar by value, so the resolver must
    # update all bound copies as part of the atomic implementation selection.
    _fast.BI_GDN_SOLVE_TRIL_DECODE = slot_direct
    _incr.BI_GDN_SOLVE_TRIL_DECODE = slot_direct
    _heal.BI_GDN_SOLVE_TRIL_DECODE = slot_direct
    _decode.BI_GDN_DECODE_ENABLED = True
    _decode.BI_GDN_BS1_STATIC = slot_direct
    _decode.BI_GDN_DECODE_GRAPH = exact_graph
    # The modern attention backend uses the slot-direct transport for exact
    # graph replay.  Its arithmetic stages are the oracle prefill binaries;
    # unlike the retired single-bucket staging path it supports every exact
    # decode graph shape through the graph-32 ceiling.
    _fast.BI_GDN_DECODE_FAST_ENABLED = slot_direct
    _fast.BI_GDN_FUSE_SMALL_ENABLED = slot_direct
    _incr.BI_GDN_DECODE_INCR_ENABLED = exact_graph
    # Deferred writeback is a graph optimization, not part of the arithmetic.
    # Aligned extra-buffer radix snapshots need the live recurrent state at
    # every track boundary, so use the same incremental runner with immediate
    # writeback when prefix reuse is enabled.  ServerArgs rejects no_buffer:
    # it can expose arbitrary-prefix checkpoints without the private exact-GDN
    # boundary and partial-row buffers needed to resume them.
    _incr.BI_GDN_INCR_DEFER_ENABLED = _defer_incremental_state_writeback(
        server_args, exact_graph=exact_graph
    )
    _incr.BI_GDN_VNEW_SLIM_ENABLED = exact_graph
    _heal.BI_GDN_LAZY_HEAL_ENABLED = exact_graph
    _gemm_configs._force_bi_gemm_config_table(is_moe)
    _norm_gated.set_gdn_norm_rows_per_block_pin(4)
    # The fused Wave-3 GEMM/router/head mechanisms remain HELD behind their
    # own live promotion: this re-promotion restores exactly the cached-row
    # incremental GDN set the live trainer-to-sampler gate re-qualified on
    # the fixed solve kernel. The promoted tuple's fused ordered combine was
    # structurally superseded by the canonical contributor fold.
    _tiera.set_tiera_enabled(False)
    _bi_ops_ext.set_router_renorm_fused_enabled(False)
    _bi_ops_ext.set_bi_head_fastpath_enabled(False)

    logger.info(
        "Exact Qwen3.5-family zero-K3 serving resolved: BI GDN "
        "prefill/rescan decode%s, rows-per-block pin, contract lm-head + "
        "decode rescore; rmsnorm_family=%s; moe_fold=%s; resolved tuple=%s",
        (
            ", cached-row incremental graph decode + batched cache warm + exact "
            "rescan eager fallback, decode-scheduled solve, fused small stages; "
            "Tier-A BI-GEMM configs, fused router renorm, and head fastpath "
            "held behind live promotion"
            if is_moe
            else " (conservative eager tuple; MoE Wave-3 fast paths disabled)"
        ),
        rmsnorm_family,
        "none",
        "physical-pp,stage-local-owners,graph-or-eager,radix-optional",
    )
    _applied = True


__all__ = [
    "QWEN35_DENSE_ARCHS",
    "QWEN35_EXACT_ARCHS",
    "QWEN35_MOE_ARCHS",
    "QWEN35_REQUIRED_BI_OPS",
]
