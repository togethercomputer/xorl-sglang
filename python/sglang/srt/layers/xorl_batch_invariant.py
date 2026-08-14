import logging
from typing import Any, Callable, List, Literal

import torch

from sglang.srt.batch_invariant_ops import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNormFamily,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob_from_logits,
    exact_temperature_scale_fp32_logits,
    head_v2_full_logits_with_lse,
    head_v2_selected_logprob_from_logits,
)
from sglang.srt.layers.exact_sampling_transforms import (
    exact_masked_logits,
    exact_sampling_identity_rows,
    exact_seeded_gumbel_sample,
    exact_selected_logprob_partitioned_from_support,
)

XorlBiFamily = Literal["v1", "v2"]
XorlGlm52NormSite = Literal["q_a", "kv_a", "input", "post_attention", "final"]
BI_FAMILIES_V2_CONTRACT = "xorl_batch_invariant_families_v2"
XORL_GLM52_REQUIRED_BI_OPS = ("addmm", "bmm", "log_softmax", "mean", "mm")
_CONTRACT_PLAN_LOGGED = False
logger = logging.getLogger(__name__)


_FORCED_FAMILY: XorlBiFamily | None = None


def force_xorl_bi_family(family: XorlBiFamily) -> None:
    """Pin the numerical family internally for an architecture resolver.

    The exact path selects the best certified family structurally; once
    forced, no environment variable is consulted.
    """
    global _FORCED_FAMILY
    if family not in ("v1", "v2"):
        raise ValueError(f"Unknown batch-invariant family {family!r}")
    _FORCED_FAMILY = family


def resolve_xorl_bi_family() -> XorlBiFamily:
    """Resolve one numerical family for every strict trainer/sampler site."""
    if _FORCED_FAMILY is not None:
        return _FORCED_FAMILY
    return "v2"


def resolve_or_validate_xorl_bi_family(family: str | None) -> XorlBiFamily:
    """Reject a caller override that disagrees with the process-wide family."""
    resolved = resolve_xorl_bi_family()
    if family is None:
        return resolved
    if family not in ("v1", "v2"):
        raise RuntimeError(f"Unknown XORL batch-invariant family: {family!r}.")
    if family != resolved:
        raise RuntimeError(
            "The requested XORL batch-invariant family disagrees with the "
            f"process-wide contract: requested={family!r}, resolved={resolved!r}."
        )
    return resolved


def log_xorl_bi_contract_plan_once(
    plan_logger: logging.Logger,
    *,
    use_qk_norm: bool,
    speculative_decode: bool,
    mtp_decode: bool,
    legacy_bi_ops: tuple[str, ...],
) -> XorlBiFamily:
    """Validate and emit the serving plan before the first real forward."""
    global _CONTRACT_PLAN_LOGGED
    family = resolve_xorl_bi_family()
    validate_xorl_glm52_norm_envelope(use_qk_norm=use_qk_norm)
    if speculative_decode or mtp_decode:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract requires speculative "
            "and MTP decoding to be disabled."
        )
    if tuple(sorted(legacy_bi_ops)) != XORL_GLM52_REQUIRED_BI_OPS:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract requires exactly the "
            f"legacy BI ops {XORL_GLM52_REQUIRED_BI_OPS}, got "
            f"{tuple(sorted(legacy_bi_ops))}."
        )
    if not _CONTRACT_PLAN_LOGGED:
        plan_logger.info(
            "XORL batch-invariant numerical contract plan: family=%s "
            "family_contract=%s serving_target=xorl "
            "resolved_use_qk_norm=false speculative_decode=false mtp_decode=false "
            "legacy_bi_ops=%s glm52_bi_router=true "
            "required_peer_trainer_rmsnorm_mode=sglang_fused",
            family,
            BI_FAMILIES_V2_CONTRACT,
            ",".join(XORL_GLM52_REQUIRED_BI_OPS),
        )
        _CONTRACT_PLAN_LOGGED = True
    return family


def validate_xorl_glm52_norm_envelope(*, use_qk_norm: bool) -> None:
    if use_qk_norm:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract does not certify "
            "use_qk_norm=True; the official checkpoint has no q/k norm helper."
        )


def xorl_glm52_norm_site_family(
    site: XorlGlm52NormSite,
    *,
    layer_id: int | None = None,
) -> RMSNormFamily:
    """Map each official GLM-5.2 RMSNorm site to its serving v1 family."""
    if site in ("q_a", "kv_a"):
        return RMS_NORM_FAMILY_NO_RESIDUAL
    if site == "input":
        if layer_id is None or layer_id < 0:
            raise ValueError("A GLM-5.2 input RMSNorm site requires layer_id >= 0.")
        return (
            RMS_NORM_FAMILY_NO_RESIDUAL
            if layer_id == 0
            else RMS_NORM_FAMILY_RESIDUAL_TREE
        )
    if site in ("post_attention", "final"):
        return RMS_NORM_FAMILY_RESIDUAL_TREE
    raise ValueError(f"Unknown GLM-5.2 RMSNorm site: {site!r}.")


def validate_xorl_bi_logit_transforms(
    logit_scale: float | None,
    final_logit_softcapping: float | None,
) -> None:
    if logit_scale is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support logit_scale."
        )
    if final_logit_softcapping is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support "
            "final_logit_softcapping."
        )


def _apply_xorl_exact_lora_lm_head(
    hidden_states: torch.Tensor,
    lm_head: Any,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Apply the literal Triton LoRA head to exact FP32 base logits.

    ``ParallelLMHeadWithLoRA.apply_lora`` is the serving implementation: its A
    kernel accumulates in FP32 and stores BF16, then its B kernel rounds the
    delta to BF16 before adding it to this FP32 base-logit buffer.  Keeping the
    operation here makes the exact GLM path own the base head while reusing the
    real adapter buffers, batch metadata, kernels, scaling, and in-place add.
    """

    from sglang.srt.lora.layers import ParallelLMHeadWithLoRA  # noqa: PLC0415

    if not isinstance(lm_head, ParallelLMHeadWithLoRA):
        if hasattr(lm_head, "set_lora") or hasattr(lm_head, "apply_lora"):
            raise RuntimeError(
                "The XORL exact active-LoRA LM-head contract accepts only the "
                "real ParallelLMHeadWithLoRA wrapper."
            )
        return base_logits

    backend = getattr(lm_head, "lora_backend", None)
    if getattr(backend, "name", None) != "triton":
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires the Triton "
            "LoRA backend."
        )
    lora_active = getattr(lm_head, "lora_active", None)
    if not isinstance(lora_active, bool):
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires a boolean "
            "lora_active state from ParallelLMHeadWithLoRA."
        )
    if not lora_active:
        return base_logits
    if getattr(lm_head, "set_lora", None) is not True:
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires installed "
            "LoRA buffers for an active request."
        )
    if getattr(backend, "batch_info", None) is None:
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires prepared "
            "per-request LoRA batch metadata."
        )

    a_buffer = getattr(lm_head, "lm_head_A_buffer", None)
    b_buffer = getattr(lm_head, "lm_head_B_buffer", None)
    if not isinstance(a_buffer, torch.Tensor) or not isinstance(b_buffer, torch.Tensor):
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires live A/B "
            "memory-pool tensors."
        )
    if a_buffer.dtype != torch.bfloat16 or b_buffer.dtype != torch.bfloat16:
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires BF16 A/B "
            f"buffers, got {a_buffer.dtype} and {b_buffer.dtype}."
        )
    physical_rank = a_buffer.shape[-2] if a_buffer.ndim == 3 else -1
    expected_a_tail = (physical_rank, hidden_states.shape[-1])
    expected_b_tail = (lm_head.weight.shape[0], physical_rank)
    if (
        a_buffer.ndim != 3
        or tuple(a_buffer.shape[-2:]) != expected_a_tail
        or b_buffer.ndim != 3
        or tuple(b_buffer.shape[-2:]) != expected_b_tail
        or a_buffer.shape[0] != b_buffer.shape[0]
        or physical_rank <= 0
    ):
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires admitted "
            "physical buffers [slots,rank,hidden] and [slots,local_vocab,rank], got "
            f"A={tuple(a_buffer.shape)} and B={tuple(b_buffer.shape)}."
        )
    if not a_buffer.is_contiguous() or not b_buffer.is_contiguous():
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires contiguous "
            "physical A/B buffers."
        )

    result = lm_head.apply_lora(base_logits, hidden_states)
    if (
        result is not base_logits
        or result.dtype != torch.float32
        or tuple(result.shape) != tuple(base_logits.shape)
        or not result.is_contiguous()
    ):
        raise RuntimeError(
            "The XORL exact active-LoRA LM-head contract requires the literal "
            "B kernel to update the contiguous FP32 base-logit buffer in place."
        )
    return result


def xorl_bi_lm_head(
    hidden_states: torch.Tensor,
    lm_head: Any,
    *,
    use_fp32_lm_head: bool,
    embedding_bias: torch.Tensor | None = None,
    family: XorlBiFamily | None = None,
) -> torch.Tensor:
    if use_fp32_lm_head:
        raise RuntimeError(
            "The XORL batch-invariant target rejects --enable-fp32-lm-head: "
            "its contract consumes BF16 hidden states and BF16 weights directly "
            "and produces FP32 logits without an FP32-copy matmul."
        )
    if embedding_bias is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support an embedding bias."
        )
    if not hasattr(lm_head, "weight"):
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract requires a dense BF16 weight."
        )
    if hidden_states.dtype != torch.bfloat16 or lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract requires BF16 hidden states "
            f"and BF16 weights, got {hidden_states.dtype} and {lm_head.weight.dtype}."
        )
    if not lm_head.weight.is_contiguous():
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract requires a contiguous BF16 weight."
        )
    hidden_states = hidden_states.contiguous()
    family = resolve_or_validate_xorl_bi_family(family)
    if family == "v2":
        logits, _ = head_v2_full_logits_with_lse(hidden_states, lm_head.weight)
    else:
        logits = bi_lm_head_full_logits(hidden_states, lm_head.weight)
    if (
        logits.dtype != torch.float32
        or tuple(logits.shape) != (hidden_states.shape[0], lm_head.weight.shape[0])
        or not logits.is_contiguous()
    ):
        raise RuntimeError(
            "The XORL batch-invariant LM-head kernel returned an invalid local FP32 buffer."
        )
    return _apply_xorl_exact_lora_lm_head(hidden_states, lm_head, logits)


def xorl_bi_sample_and_score(
    logits_output: Any,
    sampling_info: Any,
    *,
    return_logprob: bool,
    top_logprobs_nums: List[int] | None,
    token_ids_logprobs: List[List[int] | None] | None,
    positions: torch.Tensor,
    sample_from_logprobs: (
        Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor] | None
    ),
    sync_token_ids: Callable[[torch.Tensor, Any], None],
    enable_deterministic: bool,
    return_original_logprob: bool,
    family: XorlBiFamily | None = None,
) -> torch.Tensor:
    """Sample and score under the exact per-row FP32 temperature contract."""
    family = resolve_or_validate_xorl_bi_family(family)
    logits = logits_output.next_token_logits
    if logits is None or logits.dtype != torch.float32:
        raise RuntimeError(
            "The XORL batch-invariant sampler requires FP32 logits from "
            "bi_lm_head_full_logits."
        )
    if not enable_deterministic or sampling_info.sampling_seed is None:
        raise RuntimeError(
            "The XORL batch-invariant sampler requires deterministic inference "
            "and a per-request sampling seed."
        )
    has_sampling_filter = (
        sampling_info.need_top_p_sampling
        or sampling_info.need_top_k_sampling
        or sampling_info.need_min_p_sampling
    )
    if sampling_info.has_custom_logit_processor:
        raise RuntimeError(
            "The XORL batch-invariant sampler does not support custom logit processors."
        )
    if (
        sampling_info.acc_additive_penalties is not None
        or sampling_info.acc_scaling_penalties is not None
        or (
            sampling_info.penalizer_orchestrator is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
        or sampling_info.grammars
        or sampling_info.grammar_mask is not None
        or sampling_info.logit_bias is not None
    ):
        raise RuntimeError(
            "The XORL batch-invariant sampler does not support penalties, "
            "grammar masks, or logit bias."
        )
    if return_original_logprob:
        raise RuntimeError(
            "The XORL batch-invariant sampler rejects SGLANG_RETURN_ORIGINAL_LOGPROB."
        )
    if (top_logprobs_nums is not None and any(x > 0 for x in top_logprobs_nums)) or (
        token_ids_logprobs is not None
        and any(token_ids is not None for token_ids in token_ids_logprobs)
    ):
        raise RuntimeError(
            "The XORL batch-invariant sampler only returns the sampled token logprob."
        )

    torch._assert_async(
        torch.isfinite(logits).all(),
        "The XORL batch-invariant sampler requires finite logits.",
    )

    transformed_logits = exact_temperature_scale_fp32_logits(
        logits,
        sampling_info.temperatures,
    )
    sampling_logits = transformed_logits
    support = None
    if has_sampling_filter:
        sampling_logits, support = exact_masked_logits(
            transformed_logits,
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
        )

    is_any_greedy = bool(
        getattr(sampling_info, "is_any_greedy", sampling_info.is_all_greedy)
    )
    greedy_rows = (
        sampling_info.top_ks <= 1
        if is_any_greedy
        else torch.zeros(
            transformed_logits.shape[0],
            dtype=torch.bool,
            device=transformed_logits.device,
        )
    )
    if sampling_info.is_all_greedy:
        # SamplingParams represents temperature=0 as temperature=1/top_k=1.
        # Select the stable first argmax directly; this avoids dividing by
        # zero while preserving SGLang's established greedy normalization.
        batch_next_token_ids = torch.argmax(transformed_logits, dim=-1)
    else:
        # Gumbel-max only depends on relative logits, so sampling directly from
        # the transformed contract logits is identical to sampling from their
        # normalized logprobs and keeps selection on the distribution we report.
        if sample_from_logprobs is None:
            batch_next_token_ids = exact_seeded_gumbel_sample(
                transformed_logits,
                sampling_info.sampling_seed,
                positions,
                support=support,
            )
        else:
            # Dependency-injection hook retained for focused sampler tests.
            batch_next_token_ids = sample_from_logprobs(
                sampling_logits,
                sampling_info,
                positions,
            )
        if is_any_greedy:
            batch_next_token_ids = torch.where(
                greedy_rows,
                torch.argmax(transformed_logits, dim=-1),
                batch_next_token_ids,
            )
    sync_token_ids(batch_next_token_ids, sampling_info)

    if return_logprob:
        if sampling_info.is_all_greedy:
            # A deterministic argmax decision has probability one under the
            # decision-time distribution, hence an exact logprob of +0.
            selected_logprobs = torch.zeros(
                transformed_logits.shape[0],
                dtype=transformed_logits.dtype,
                device=transformed_logits.device,
            )
        elif support is not None:
            identity_rows = exact_sampling_identity_rows(
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                vocab_size=transformed_logits.shape[1],
            )
            score = (
                head_v2_selected_logprob_from_logits
                if family == "v2"
                else bi_lm_head_selected_logprob_from_logits
            )
            selected_logprobs, _, _ = exact_selected_logprob_partitioned_from_support(
                transformed_logits,
                batch_next_token_ids.to(torch.int64),
                support,
                identity_rows,
                lambda native_logits, native_ids: score(
                    native_logits,
                    native_ids,
                    temperature=None,
                ),
            )
        else:
            score = (
                head_v2_selected_logprob_from_logits
                if family == "v2"
                else bi_lm_head_selected_logprob_from_logits
            )
            selected_logprobs, _, _ = score(
                transformed_logits,
                batch_next_token_ids,
                temperature=None,
            )
        if is_any_greedy and not sampling_info.is_all_greedy:
            selected_logprobs = torch.where(
                greedy_rows,
                torch.zeros_like(selected_logprobs),
                selected_logprobs,
            )
        logits_output.next_token_logprobs = selected_logprobs

    return batch_next_token_ids
